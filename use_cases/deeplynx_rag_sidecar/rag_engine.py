"""RAG engine for the DeepLynx sidecar.

Applies the same multi-step Retrieval-Augmented Generation pattern used in
``msr_digital_twin_with_rag.py`` (MSR Data Layer), adapted for DeepLynx's
data model.

Key design decisions inherited from MSR Data Layer:
* Four-tier embedding fallback: OpenAI → GitHub Models → local GPU →
  random-projection (zero external deps, works air-gapped).
* Sentence-aware chunking with overlap.
* Hybrid dense-cosine + sparse TF-IDF retrieval.
* Multi-step pipeline: query decomposition → parallel sub-query search →
  per-result extraction → final synthesis.
* Stubs for every external I/O path; unit tests make zero network calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import textwrap
import urllib.error
import urllib.request
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# ── tuneable constants ──────────────────────────────────────────────────────
_CHUNK_WORDS = 300
_CHUNK_OVERLAP = 60
_MAX_CHUNKS_PER_DOC = 200
_TOP_K_DEFAULT = 5
_EMBED_DIM_FALLBACK = 256

# ── environment config ──────────────────────────────────────────────────────
_OPENAI_KEY = os.environ.get("DEEPLYNX_OPENAI_API_KEY") or os.environ.get(
    "MSR_OPENAI_API_KEY", ""
)
_OPENAI_BASE = os.environ.get(
    "DEEPLYNX_OPENAI_BASE_URL", "https://api.openai.com/v1"
)
_OPENAI_MODEL = os.environ.get("DEEPLYNX_OPENAI_MODEL", "gpt-4o-mini")
_EMBED_MODEL = os.environ.get("DEEPLYNX_EMBED_MODEL", "text-embedding-3-small")
_GITHUB_TOKEN = os.environ.get("DEEPLYNX_GITHUB_TOKEN") or os.environ.get(
    "GITHUB_TOKEN", ""
)


# ── low-level LLM helpers ───────────────────────────────────────────────────

def _openai_chat(messages: list[dict[str, str]], base_url: str, key: str) -> str:
    """Call an OpenAI-compatible chat completions endpoint."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {"model": _OPENAI_MODEL, "messages": messages, "temperature": 0.2}
    ).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM call failed: %s", exc)
        return ""


def _call_llm(prompt: str) -> str:
    """Call the best available LLM, returning empty string on failure."""
    messages = [{"role": "user", "content": prompt}]
    if _OPENAI_KEY:
        return _openai_chat(messages, _OPENAI_BASE, _OPENAI_KEY)
    if _GITHUB_TOKEN:
        return _openai_chat(
            messages,
            "https://models.inference.ai.azure.com",
            _GITHUB_TOKEN,
        )
    return ""


# ── embedding engines ────────────────────────────────────────────────────────

def _embed_openai(texts: list[str], base_url: str, key: str) -> list[list[float]]:
    url = base_url.rstrip("/") + "/embeddings"
    payload = json.dumps({"model": _EMBED_MODEL, "input": texts}).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return [item["embedding"] for item in data["data"]]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding call failed: %s", exc)
        return []


def _random_projection_embed(text: str, dim: int = _EMBED_DIM_FALLBACK) -> list[float]:
    """Deterministic random-projection embedding (zero external deps)."""
    import struct

    seed = int(hashlib.md5(text.lower().encode()).hexdigest(), 16) % (2**32)
    # Simple lcg-based projection — reproducible across Python versions
    state = seed
    vec: list[float] = []
    for _ in range(dim):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        val = struct.unpack("f", struct.pack("I", state))[0]
        vec.append(float(val) if math.isfinite(val) else 0.0)
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Return embeddings using the best available engine."""
    if _OPENAI_KEY:
        result = _embed_openai(texts, _OPENAI_BASE, _OPENAI_KEY)
        if result:
            return result
    if _GITHUB_TOKEN:
        result = _embed_openai(
            texts, "https://models.inference.ai.azure.com", _GITHUB_TOKEN
        )
        if result:
            return result
    # Local GPU (optional import — graceful fallback)
    if os.environ.get("DEEPLYNX_USE_LOCAL_GPU", "").lower() == "true":
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model_name = os.environ.get(
                "DEEPLYNX_LOCAL_EMBED_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            )
            model = SentenceTransformer(model_name)
            return model.encode(texts).tolist()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Local GPU embedding failed: %s", exc)
    # Fallback: deterministic random projection
    return [_random_projection_embed(t) for t in texts]


# ── chunking ─────────────────────────────────────────────────────────────────

def _sentence_chunk(text: str) -> list[str]:
    """Split *text* into sentence-aware chunks of roughly _CHUNK_WORDS words."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0
    overlap_buf: list[str] = []

    for sent in sentences:
        words = sent.split()
        if buf_words + len(words) > _CHUNK_WORDS and buf:
            chunks.append(" ".join(buf))
            if len(chunks) >= _MAX_CHUNKS_PER_DOC:
                break
            # carry over overlap
            overlap_words: list[str] = []
            for s in reversed(buf):
                overlap_words = s.split() + overlap_words
                if len(overlap_words) >= _CHUNK_OVERLAP:
                    break
            buf = [" ".join(overlap_words)] if overlap_words else []
            buf_words = len(overlap_words)
        buf.append(sent)
        buf_words += len(words)

    if buf and len(chunks) < _MAX_CHUNKS_PER_DOC:
        chunks.append(" ".join(buf))
    return chunks or [text[:2000]]


# ── TF-IDF helpers ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


# ── KnowledgeStore ───────────────────────────────────────────────────────────

class KnowledgeStore:
    """In-memory KB: chunks, dense embeddings, TF-IDF index."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.embeddings: list[list[float]] = []
        self._idf: dict[str, float] = {}
        self._tf_vecs: list[dict[str, float]] = []

    # ── ingest ────────────────────────────────────────────────────────────────

    def add_document(self, text: str, source_id: str) -> int:
        """Chunk *text*, embed, and add to KB. Returns number of chunks added."""
        new_chunks = _sentence_chunk(text)
        new_embeds = _get_embeddings(new_chunks)
        if len(new_embeds) != len(new_chunks):
            new_embeds = [_random_projection_embed(c) for c in new_chunks]

        self.chunks.extend(new_chunks)
        self.sources.extend([source_id] * len(new_chunks))
        self.embeddings.extend(new_embeds)
        self._rebuild_tfidf()
        return len(new_chunks)

    def _rebuild_tfidf(self) -> None:
        n = len(self.chunks)
        if n == 0:
            return
        df: Counter[str] = Counter()
        tfs: list[dict[str, float]] = []
        for chunk in self.chunks:
            tokens = _tokenize(chunk)
            tf = Counter(tokens)
            total = len(tokens) or 1
            tfs.append({t: c / total for t, c in tf.items()})
            df.update(tf.keys())
        self._idf = {t: math.log((n + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}
        self._tf_vecs = [
            {t: tf_val * self._idf.get(t, 1.0) for t, tf_val in tf.items()}
            for tf in tfs
        ]

    # ── search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = _TOP_K_DEFAULT) -> list[dict[str, Any]]:
        """Hybrid dense + sparse search. Returns list of result dicts."""
        if not self.chunks:
            return []
        q_embed = _get_embeddings([query])[0]
        dense_scores = [_cosine(q_embed, e) for e in self.embeddings]

        q_tokens = _tokenize(query)
        q_tf = Counter(q_tokens)
        q_total = len(q_tokens) or 1
        q_tfidf = {t: (c / q_total) * self._idf.get(t, 1.0) for t, c in q_tf.items()}
        sparse_scores = [
            sum(q_tfidf.get(t, 0.0) * v for t, v in vec.items())
            for vec in self._tf_vecs
        ]

        # Normalise and combine
        max_d = max(dense_scores) or 1e-9
        max_s = max(sparse_scores) or 1e-9
        combined = [
            (i, 0.6 * dense_scores[i] / max_d + 0.4 * sparse_scores[i] / max_s)
            for i in range(len(self.chunks))
        ]
        combined.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "chunk": self.chunks[i],
                "source": self.sources[i],
                "score": score,
            }
            for i, score in combined[:top_k]
        ]


# ── DeepLynxRAG ───────────────────────────────────────────────────────────────

class DeepLynxRAG:
    """Multi-step RAG over a DeepLynx project catalog.

    Usage::

        rag = DeepLynxRAG()
        rag.ingest_project_records(container_id="my-project")
        answer = rag.answer("Which welds used non-conforming filler material?")
    """

    def __init__(self) -> None:
        self.kb = KnowledgeStore()
        self._ingested_sources: set[str] = set()

    # ── ingestion ─────────────────────────────────────────────────────────────

    def ingest_text(self, text: str, source_id: str) -> int:
        """Add arbitrary text to the KB.

        Args:
            text: Document text.
            source_id: Unique source identifier for provenance.

        Returns:
            Number of chunks created.
        """
        if source_id in self._ingested_sources:
            return 0
        n = self.kb.add_document(text, source_id)
        if n:
            self._ingested_sources.add(source_id)
        return n

    def ingest_project_records(
        self,
        container_id: str,
        datasource_id: str | None = None,
        limit: int = 200,
    ) -> int:
        """Fetch and index records from a DeepLynx project.

        Args:
            container_id: DeepLynx container (project) ID.
            datasource_id: Optional data source filter.
            limit: Max records to ingest.

        Returns:
            Total number of chunks created.
        """
        from deeplynx_client import get_records  # local import for testability

        records = get_records(container_id, datasource_id, limit=limit)
        total = 0
        for rec in records:
            rec_id = str(rec.get("id", ""))
            rec_type = str(rec.get("type", "record"))
            props = rec.get("properties", {})
            # Combine all string property values into a searchable text blob
            text_parts = [f"[{rec_type}] ID: {rec_id}"]
            for k, v in props.items():
                if isinstance(v, str) and v.strip():
                    text_parts.append(f"{k}: {v}")
            text = "\n".join(text_parts)
            source_id = f"deeplynx:{container_id}:{rec_id}"
            total += self.ingest_text(text, source_id)
        logger.info("Ingested %d chunks from %d records", total, len(records))
        return total

    # ── query pipeline ────────────────────────────────────────────────────────

    def _decompose_query(self, question: str) -> list[str]:
        """Break a complex question into targeted sub-queries via LLM."""
        prompt = textwrap.dedent(f"""
            You are helping search an engineering project data catalog.
            Break this question into at most 4 focused sub-queries that will
            each retrieve a different relevant set of records.
            Return ONLY a JSON array of strings.

            Question: {question}
        """).strip()
        raw = _call_llm(prompt)
        try:
            queries = json.loads(raw)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return queries[:4]
        except (json.JSONDecodeError, ValueError):
            pass
        return [question]

    def _extract_sub_answer(self, chunk: str, sub_query: str) -> str:
        """Distil a chunk into a focused partial answer for *sub_query*."""
        prompt = textwrap.dedent(f"""
            Extract information relevant to the query from the context below.
            Be concise (≤3 sentences). If the context does not answer the query,
            reply with "Not relevant."

            Query: {sub_query}
            Context: {chunk[:1500]}
        """).strip()
        return _call_llm(prompt) or chunk[:300]

    def _synthesise(self, question: str, sub_answers: list[str]) -> str:
        """Combine sub-answers into a final response."""
        joined = "\n\n".join(f"[{i+1}] {a}" for i, a in enumerate(sub_answers))
        prompt = textwrap.dedent(f"""
            Synthesise the following partial answers into a single comprehensive
            response to the question. Cite source numbers like [1], [2] where
            relevant. Be factual and concise.

            Question: {question}

            Partial answers:
            {joined}
        """).strip()
        return _call_llm(prompt) or joined

    def answer(
        self, question: str, top_k: int = _TOP_K_DEFAULT, container_id: str = ""
    ) -> dict[str, Any]:
        """Run the full multi-step RAG pipeline.

        Args:
            question: Natural-language question about the project data.
            top_k: Number of chunks to retrieve per sub-query.
            container_id: Optional DeepLynx container ID shown in provenance.

        Returns:
            Dict with keys ``answer``, ``sources`` (list of source IDs),
            ``sub_queries``, and ``chunks_used`` (count).
        """
        if not self.kb.chunks:
            return {
                "answer": "No records have been indexed yet. Call ingest_project_records() first.",
                "sources": [],
                "sub_queries": [],
                "chunks_used": 0,
            }

        llm_available = bool(_call_llm("ping"))
        sub_queries = self._decompose_query(question) if llm_available else [question]
        # Fallback: if LLM unavailable, skip decomposition
        if not any(sub_queries):
            sub_queries = [question]

        all_results: list[dict[str, Any]] = []
        seen_chunks: set[str] = set()
        for sq in sub_queries:
            for result in self.kb.search(sq, top_k=top_k):
                key = result["chunk"][:100]
                if key not in seen_chunks:
                    seen_chunks.add(key)
                    result["sub_query"] = sq
                    all_results.append(result)

        sources = list(dict.fromkeys(r["source"] for r in all_results))

        if not llm_available:
            # No LLM: return raw context
            context = "\n\n---\n\n".join(r["chunk"] for r in all_results[:top_k])
            return {
                "answer": context,
                "sources": sources,
                "sub_queries": sub_queries,
                "chunks_used": len(all_results),
            }

        # Extract partial answers per chunk
        sub_answers = [
            self._extract_sub_answer(r["chunk"], r["sub_query"])
            for r in all_results[:top_k * 2]
        ]
        sub_answers = [a for a in sub_answers if "not relevant" not in a.lower()]

        final_answer = (
            self._synthesise(question, sub_answers)
            if sub_answers
            else "No relevant information found in the indexed records."
        )
        return {
            "answer": final_answer,
            "sources": sources,
            "sub_queries": sub_queries,
            "chunks_used": len(all_results),
        }
