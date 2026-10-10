from __future__ import annotations

import httpx
import pytest
import respx

from mcp_agent_database.retrieval import (
    BM25,
    DocumentIndex,
    OllamaEmbedder,
    chunk_markdown,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_drops_stopwords_and_punctuation() -> None:
    assert tokenize("What is the Refund window, for Laptops?") == ["refund", "window", "laptops"]


def test_chunks_keep_heading_trail() -> None:
    text = (
        "# Policy\nintro\n\n## Returns\nThirty days.\n\n"
        "### Headphones\nFourteen days.\n```\n# not a heading\n```"
    )
    chunks = chunk_markdown(text)
    headings = [h for h, _ in chunks]
    assert headings == ["Policy", "Policy > Returns", "Policy > Returns > Headphones"]
    assert "# not a heading" in chunks[-1][1]


def test_long_sections_are_split_with_overlap() -> None:
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    chunks = chunk_markdown(f"# Big\n{body}", max_chars=500, overlap=50)
    assert len(chunks) > 3
    assert all(len(c) <= 500 + 50 + 2 for _, c in chunks)


def test_bm25_prefers_matching_and_rare_terms() -> None:
    corpus = [
        tokenize(t) for t in ["refund window thirty days", "shipping costs", "refund refund laptops fee"]
    ]
    scores = BM25(corpus).scores(tokenize("laptops refund"))
    assert scores.index(max(scores)) == 2
    assert scores[1] == 0


def test_rrf_rewards_agreement() -> None:
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert max(fused, key=fused.get) in {1, 2}
    assert fused[2] > fused[3]
    assert fused[4] < fused[1]


def test_index_finds_relevant_policy(settings) -> None:
    index = DocumentIndex(settings.docs_dir).build()
    assert index.mode == "bm25"
    top = index.search("how many days do enterprise customers have to return items")[0]
    assert top.chunk.doc_id == "refund-policy"
    assert "Enterprise" in top.chunk.heading


@respx.mock
def test_hybrid_mode_with_embeddings(settings) -> None:
    def fake_embed(request: httpx.Request) -> httpx.Response:
        import json

        texts = json.loads(request.content)["input"]
        # 2-d "embedding": does the text mention shipping?
        return httpx.Response(
            200, json={"embeddings": [[1.0, 0.0] if "ship" in t.lower() else [0.0, 1.0] for t in texts]}
        )

    respx.post("http://ollama.test/api/embed").mock(side_effect=fake_embed)
    index = DocumentIndex(settings.docs_dir, embedder=OllamaEmbedder("nomic", "http://ollama.test")).build()
    assert index.mode == "hybrid"
    hits = index.search("shipping")
    assert hits[0].chunk.doc_id == "shipping-policy"
    assert hits[0].dense_rank is not None


@respx.mock
def test_falls_back_to_bm25_when_ollama_is_down(settings) -> None:
    respx.post("http://ollama.test/api/embed").mock(side_effect=httpx.ConnectError("down"))
    index = DocumentIndex(settings.docs_dir, embedder=OllamaEmbedder("nomic", "http://ollama.test")).build()
    assert index.mode == "bm25"
    assert index.search("refund")


@pytest.mark.parametrize("query", ["", "zzzz qqqq"])
def test_no_results_for_unknown_terms(settings, query: str) -> None:
    assert DocumentIndex(settings.docs_dir).build().search(query) == []
