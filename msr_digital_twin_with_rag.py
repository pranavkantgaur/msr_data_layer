"""
MSR Digital Twin with RAG (Retrieval-Augmented Generation)

Combines the MSR digital twin's live sensor data (via the MCP client) with
a simple retrieval-augmented generation pipeline so that an LLM can answer
natural-language questions that require both real-time plant data AND
knowledge from technical documents (manuals, safety guides, etc.).

Architecture
------------
1. **Document store** – markdown/text files in ``./docs/`` are split into
   chunks and stored with simple TF-IDF vectors.
2. **Retriever** – on each question the top-k most relevant chunks are
   retrieved using cosine similarity.
3. **Digital twin context** – the current reactor state is fetched from the
   MCP server and appended to the prompt.
4. **LLM** – the enriched prompt is sent to an OpenAI-compatible chat
   endpoint (configurable via environment variables).

Environment Variables
---------------------
MSR_OPENAI_BASE_URL   OpenAI-compatible API base URL
                      (default: https://api.openai.com/v1)
MSR_OPENAI_API_KEY    API key (required when using a real LLM backend)
MSR_OPENAI_MODEL      Model name (default: gpt-4o-mini)
MSR_DOCS_DIR          Path to the directory containing reference documents
                      (default: ./docs)
"""

from __future__ import annotations

import math
import os
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

from msr_digital_twin_client import MSRDigitalTwinClient


# ---------------------------------------------------------------------------
# Simple TF-IDF document store
# ---------------------------------------------------------------------------

class DocumentStore:
    """
    Minimal in-memory document store with TF-IDF retrieval.

    Suitable for small corpora (< 10 000 chunks).  For larger deployments
    replace with a vector database such as ChromaDB or Qdrant.
    """

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._sources: list[str] = []
        self._tfidf: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_document(self, text: str, source: str = "", chunk_size: int = 400) -> int:
        """
        Split *text* into overlapping chunks and add them to the store.

        Returns the number of chunks added.
        """
        tokens = _tokenize(text)
        step = chunk_size // 2
        added = 0
        for start in range(0, max(1, len(tokens) - chunk_size + 1), step):
            chunk_tokens = tokens[start: start + chunk_size]
            chunk_text = " ".join(chunk_tokens)
            self._chunks.append(chunk_text)
            self._sources.append(source)
            added += 1
        self._rebuild_idf()
        return added

    def load_directory(self, docs_dir: str | Path) -> int:
        """Load all ``.md`` and ``.txt`` files from *docs_dir*."""
        docs_dir = Path(docs_dir)
        if not docs_dir.is_dir():
            return 0
        total = 0
        paths = sorted(docs_dir.rglob("*.md")) + sorted(docs_dir.rglob("*.txt"))
        for path in sorted(set(paths)):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                total += self.add_document(text, source=str(path))
            except OSError:
                continue
        return total

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Return the *top_k* most relevant chunks for *query*."""
        if not self._chunks:
            return []
        q_vec = self._tfidf_vector(query)
        scores: list[tuple[float, int]] = []
        for idx, chunk_vec in enumerate(self._tfidf):
            score = _cosine_similarity(q_vec, chunk_vec)
            scores.append((score, idx))
        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            results.append(
                {
                    "text": self._chunks[idx],
                    "source": self._sources[idx],
                    "score": round(score, 4),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_idf(self) -> None:
        df: dict[str, int] = {}
        for chunk in self._chunks:
            for token in set(_tokenize(chunk)):
                df[token] = df.get(token, 0) + 1
        n = len(self._chunks)
        self._idf = {
            token: math.log((n + 1) / (count + 1)) + 1
            for token, count in df.items()
        }
        self._tfidf = [self._tfidf_vector(c) for c in self._chunks]

    def _tfidf_vector(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) or 1
        return {
            token: (count / total) * self._idf.get(token, 1.0)
            for token, count in tf.items()
        }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

class MSRDigitalTwinRAG:
    """
    Retrieval-Augmented Generation over the MSR digital twin.

    Parameters
    ----------
    docs_dir : str | Path | None
        Directory of reference documents to load into the store.
        If *None* the ``MSR_DOCS_DIR`` environment variable is used,
        falling back to ``./docs``.
    """

    def __init__(self, docs_dir: str | Path | None = None) -> None:
        self._store = DocumentStore()
        _docs = docs_dir or os.environ.get("MSR_DOCS_DIR", "./docs")
        loaded = self._store.load_directory(_docs)
        if loaded:
            print(f"[RAG] Loaded {loaded} document chunks from '{_docs}'.")
        else:
            print(f"[RAG] No documents found in '{_docs}'. RAG context will be empty.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_document(self, text: str, source: str = "") -> int:
        """Programmatically add a document to the store."""
        return self._store.add_document(text, source=source)

    def answer(self, question: str, top_k: int = 4) -> str:
        """
        Answer *question* by combining live reactor data with retrieved
        document context.

        If the OpenAI API key is set the answer will be generated by the
        configured LLM; otherwise a structured context summary is returned
        so the caller can pass it to their own LLM.
        """
        # 1. Retrieve relevant document chunks
        chunks = self._store.retrieve(question, top_k=top_k)
        doc_context = self._format_chunks(chunks)

        # 2. Fetch live reactor state
        reactor_context = self._fetch_reactor_context()

        # 3. Build prompt
        prompt = self._build_prompt(question, doc_context, reactor_context)

        # 4. Call LLM (if credentials available)
        api_key = os.environ.get("MSR_OPENAI_API_KEY", "")
        if api_key:
            return self._call_llm(prompt, api_key)

        # Return the enriched prompt as a plain string when no LLM is configured
        return (
            "[No LLM configured – set MSR_OPENAI_API_KEY to enable generation]\n\n"
            + prompt
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_reactor_context(self) -> str:
        try:
            with MSRDigitalTwinClient() as client:
                status = client.get_reactor_status()
                alarms = client.get_active_alarms()
            lines = ["Current reactor state:"]
            for key, value in status.items():
                lines.append(f"  {key}: {value}")
            if alarms.get("alarm_count", 0) > 0:
                lines.append(f"\nActive alarms ({alarms['alarm_count']}):")
                for alarm in alarms["alarms"]:
                    lines.append(f"  [{alarm['severity']}] {alarm['alarm_id']}: "
                                 f"{alarm['sensor']} = {alarm['value']}")
            else:
                lines.append("\nNo active alarms.")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            return f"[Could not fetch reactor state: {exc}]"

    @staticmethod
    def _format_chunks(chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "[No relevant documents found]"
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            text = textwrap.shorten(chunk["text"], width=600, placeholder="…")
            parts.append(f"[Document {i} | source: {source}]\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def _build_prompt(question: str, doc_context: str, reactor_context: str) -> str:
        return textwrap.dedent(f"""\
            You are an expert assistant for a Molten Salt Reactor (MSR) digital twin.
            Use the following information to answer the operator's question.

            ## Live Reactor Data
            {reactor_context}

            ## Reference Documents
            {doc_context}

            ## Question
            {question}

            ## Answer
        """)

    @staticmethod
    def _call_llm(prompt: str, api_key: str) -> str:
        """Send the prompt to an OpenAI-compatible endpoint and return the answer."""
        try:
            import urllib.request  # stdlib only – no extra deps

            base_url = os.environ.get("MSR_OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.environ.get("MSR_OPENAI_MODEL", "gpt-4o-mini")
            payload = json_encode(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1024,
                }
            )
            req = urllib.request.Request(
                f"{base_url}/chat/completions",
                data=payload.encode(),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json_decode(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            return f"[LLM call failed: {exc}]\n\nPrompt was:\n{prompt}"


# ---------------------------------------------------------------------------
# Thin JSON helpers (avoid importing json twice at module level)
# ---------------------------------------------------------------------------

import json as _json  # noqa: E402  (import after class defs for clarity)


def json_encode(obj: Any) -> str:
    return _json.dumps(obj)


def json_decode(s: str) -> Any:
    return _json.loads(s)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "What is the current core temperature and is it within safe limits?"
    )
    print(f"Question: {question}\n")
    rag = MSRDigitalTwinRAG()
    answer = rag.answer(question)
    print(answer)
