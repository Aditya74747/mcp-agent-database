"""Small, dependency-free retrieval engine used by the docs MCP server.

* Heading-aware chunking for Markdown (falls back to paragraph packing for .txt / .pdf).
* BM25 (Okapi) lexical search, implemented here so every step is explainable.
* Optional dense search with Ollama embeddings, merged with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS_TEXT = (
    "a an and are as at be by can for from has have how i in is it its of on or our that the this to "
    "was we what when where which who will with you your do does"
)
_STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    heading: str
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    path: Path
    text: str
    chunks: list[Chunk] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Loading and chunking
# ---------------------------------------------------------------------------------------------
def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def _pack(paragraphs: Sequence[str], max_chars: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into chunks of at most ``max_chars``, with a small tail overlap."""
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        while len(para) > max_chars:  # split very long paragraphs
            chunks.append(para[:max_chars])
            para = para[max_chars - overlap :]
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def chunk_markdown(text: str, *, max_chars: int = 900, overlap: int = 120) -> list[tuple[str, str]]:
    """Split on Markdown headings so each chunk keeps its section heading as context.

    Returns ``(heading_path, chunk_text)`` pairs, e.g. ``("Refund Policy > Enterprise", "...")``.
    """
    sections: list[tuple[list[str], list[str]]] = [([], [])]
    trail: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            level = len(match.group(1))
            trail = [*trail[: level - 1], match.group(2).strip()]
            sections.append((list(trail), []))
        else:
            sections[-1][1].append(line)

    out: list[tuple[str, str]] = []
    for headings, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        heading = " > ".join(headings)
        for piece in _pack(re.split(r"\n\s*\n", body), max_chars, overlap):
            out.append((heading, piece))
    return out


def load_documents(root: Path, *, max_chars: int = 900, overlap: int = 120) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file()):
        text = read_text(path)
        doc_id = path.relative_to(root).with_suffix("").as_posix()
        title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem.replace("-", " ").title()
        doc = Document(doc_id=doc_id, title=title, path=path, text=text)
        for i, (heading, piece) in enumerate(chunk_markdown(text, max_chars=max_chars, overlap=overlap)):
            doc.chunks.append(Chunk(f"{doc_id}#{i}", doc_id, title, heading or title, piece))
        docs.append(doc)
    return docs


# ---------------------------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------------------------
class BM25:
    """Okapi BM25 over pre-tokenized documents. ``k1`` and ``b`` use the textbook defaults."""

    def __init__(self, corpus: Sequence[Sequence[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tf = [Counter(doc) for doc in corpus]
        self.lengths = [len(doc) for doc in corpus]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        n = len(corpus)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: Sequence[str]) -> list[float]:
        out = []
        for counts, length in zip(self.tf, self.lengths, strict=True):
            score = 0.0
            for term in query:
                freq = counts.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
                score += self.idf[term] * freq * (self.k1 + 1) / denom
            out.append(score)
        return out


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], *, k: int = 60) -> dict[int, float]:
    """Combine several ranked lists of item indices: score = sum(1 / (k + rank))."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout_s: float = 60.0) -> None:
        self.model = model
        self.client = httpx.Client(base_url=base_url, timeout=timeout_s)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            resp = self.client.post(
                "/api/embed", json={"model": self.model, "input": list(texts[start : start + 32])}
            )
            resp.raise_for_status()
            vectors.extend(resp.json()["embeddings"])
        return vectors


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
    lexical_rank: int | None
    dense_rank: int | None


class DocumentIndex:
    def __init__(self, root: Path, *, embedder: OllamaEmbedder | None = None) -> None:
        self.root = root
        self.embedder = embedder
        self.documents: dict[str, Document] = {}
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] | None = None
        self.mode = "bm25"
        self._bm25 = BM25([])

    def build(self) -> DocumentIndex:
        docs = load_documents(self.root)
        self.documents = {d.doc_id: d for d in docs}
        self.chunks = [c for d in docs for c in d.chunks]
        # Index the heading together with the body: headings carry a lot of signal.
        self._bm25 = BM25([tokenize(f"{c.title} {c.heading} {c.text}") for c in self.chunks])
        self.vectors, self.mode = None, "bm25"
        if self.embedder and self.chunks:
            try:
                self.vectors = self.embedder.embed([f"{c.heading}\n{c.text}" for c in self.chunks])
                self.mode = "hybrid"
            except httpx.HTTPError as err:
                log.warning("Embeddings unavailable (%s); falling back to BM25 only.", err)
        log.info("Indexed %d documents / %d chunks (%s)", len(docs), len(self.chunks), self.mode)
        return self

    def search(self, query: str, top_k: int = 5, candidates: int = 30) -> list[SearchHit]:
        if not self.chunks:
            return []
        lexical = self._bm25.scores(tokenize(query))
        lex_rank = [i for i in sorted(range(len(lexical)), key=lambda i: -lexical[i]) if lexical[i] > 0][
            :candidates
        ]
        rankings = [lex_rank]
        dense_rank: list[int] = []
        if self.vectors is not None and self.embedder is not None:
            try:
                qv = self.embedder.embed([query])[0]
                sims = [cosine(qv, v) for v in self.vectors]
                dense_rank = sorted(range(len(sims)), key=lambda i: -sims[i])[:candidates]
                rankings.append(dense_rank)
            except httpx.HTTPError as err:
                log.warning("Query embedding failed (%s); using BM25 only.", err)
        fused = reciprocal_rank_fusion(rankings)
        ordered = sorted(fused, key=lambda i: -fused[i])[:top_k]
        lex_pos = {idx: r for r, idx in enumerate(lex_rank, start=1)}
        dense_pos = {idx: r for r, idx in enumerate(dense_rank, start=1)}
        return [
            SearchHit(self.chunks[i], round(fused[i], 5), lex_pos.get(i), dense_pos.get(i)) for i in ordered
        ]
