"""
MSR Data Layer – Enhanced RAG (Retrieval-Augmented Generation) Pipeline

Provides a knowledge base and multi-step RAG pipeline for the MSR data layer,
supporting design, construction, and operations of Molten Salt Reactor systems.

The pipeline reads live plant data via the MCP data layer client and combines
it with reference documents to answer operator and agent queries.

Architecture
------------
**Ingestion pipeline**

1. Sentence-aware text chunking with configurable overlap.
2. Dense vector embeddings via:
   - Local GPU models via sentence-transformers (when ``MSR_USE_LOCAL_GPU=true``), or
   - OpenAI-compatible Embeddings API (when ``MSR_OPENAI_API_KEY`` is set), or
   - Random-projection engine (numpy-based, zero external dependencies).
3. Source insights – LLM-generated summary, topics, and key facts for each
   ingested document (inspired by open-notebook's source transformation).
4. Persistent knowledge base: chunks, embeddings, and insights are saved to
   ``MSR_KB_DIR`` (default ``./kb_store``) so re-ingestion is not needed
   on restart.

**Retrieval pipeline (multi-step, inspired by open-notebook ask.py)**

1. *Query decomposition* – the LLM breaks the question into ≤ 5 targeted
   sub-queries, each with specific extraction instructions.
2. *Parallel search* – all sub-queries execute concurrently; each uses
   hybrid retrieval (dense cosine + sparse TF-IDF).
3. *Sub-answer extraction* – the LLM distils each search result into a
   focused partial answer.
4. *Final synthesis* – the LLM combines all partial answers and live
   plant data into a comprehensive final answer.

Environment Variables
---------------------
MSR_OPENAI_API_KEY    API key for LLM + embeddings
                      (if unset, random-projection embedding and no LLM)
MSR_OPENAI_BASE_URL   OpenAI-compatible API base URL
                      (default: https://api.openai.com/v1)
MSR_OPENAI_MODEL      Chat model (default: gpt-4o-mini)
MSR_EMBED_MODEL       Embedding model (default: text-embedding-3-small)
MSR_DOCS_DIR          Reference documents directory (default: ./docs)
MSR_KB_DIR            Persistent knowledge-base directory (default: ./kb_store)
MSR_USE_LOCAL_GPU     Set to ``true`` to use local GPU models instead of the
                      OpenAI API (requires sentence-transformers + transformers)
MSR_LOCAL_EMBED_MODEL HuggingFace model ID for local embeddings
                      (default: sentence-transformers/all-MiniLM-L6-v2)
MSR_LOCAL_LLM_MODEL   HuggingFace model ID for local text generation
                      (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
MSR_HF_CACHE_DIR      HuggingFace model cache directory
                      (default: /tmp/hf_cache)
MSR_PLANT_DATA_URL    URL of external plant data REST API (optional; when
                      unset, the development stub is used for live data reads)

Document Sources
----------------
In addition to local files in ``MSR_DOCS_DIR``, the knowledge base can be
populated from three external sources (see ``msr_kb_sources.py``):

* **Static source** – ``pranavkantgaur/msr-archive`` GitHub repository:
  OCR transcriptions of historical ORNL Molten Salt Reactor reports.
  Fetched via :meth:`MSRDigitalTwinRAG.load_msr_archive`.

* **Dynamic source** – OpenAlex academic papers API:
  Papers matching "molten salt reactors experimental data" plus a targeted
  TMSR-LF1 query (SINAP, China).
  Fetched via :meth:`MSRDigitalTwinRAG.update_openalex`.

* **Plant operational data** – real-time data pushed by operators or agents
  via :meth:`MSRDigitalTwinRAG.add_document` or the ``/data/ingest`` endpoint.
  Includes sensor snapshots, event logs, and maintenance reports.

All sources maintain state files so only new documents are re-ingested.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import re
import textwrap
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np  # type: ignore[import-untyped]
    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMPY_AVAILABLE = False

try:
    import torch  # type: ignore[import-untyped]
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from msr_digital_twin_client import MSRDigitalTwinClient


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EMBED_DIM = 256         # dense projection dimension (random-projection engine)
_VOCAB_SIZE = 16_384     # hash-based vocabulary size
_CHUNK_WORDS = 300       # target chunk size in words
_CHUNK_OVERLAP = 60      # overlap between consecutive chunks in words
_MAX_CHUNKS = 200        # per-document safety limit


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _chunk_text(
    text: str,
    chunk_words: int = _CHUNK_WORDS,
    overlap: int = _CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping word-based chunks, preferring sentence
    boundaries.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        words = sentence.split()
        if len(current) + len(words) > chunk_words and current:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap > 0 else []
        current.extend(words)
        if len(current) >= chunk_words:
            chunks.append(" ".join(current[:chunk_words]))
            current = current[chunk_words - overlap :]

    if current:
        chunks.append(" ".join(current))

    result = [c for c in chunks if c.strip()]
    return result[:_MAX_CHUNKS] if result else [text[:2000]]


def _cosine_dense(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if _NUMPY_AVAILABLE:
        av = np.array(a, dtype=np.float32)
        bv = np.array(b, dtype=np.float32)
        na, nb = float(np.linalg.norm(av)), float(np.linalg.norm(bv))
        return float(np.dot(av, bv) / (na * nb)) if na > 0 and nb > 0 else 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# ---------------------------------------------------------------------------
# Embedding engines
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """Abstract base class for computing dense text embeddings."""

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class RandomProjectionEmbeddingEngine(EmbeddingEngine):
    """
    Deterministic random-projection embedding engine.

    Builds a TF-IDF sparse vector over a hash-bucketed vocabulary, then
    projects it to a fixed-dimensional dense vector using a seeded Gaussian
    random projection matrix (Johnson–Lindenstrauss style).  No external
    dependencies beyond numpy.

    IDF weights improve with each call to :meth:`update_idf`.
    """

    def __init__(
        self,
        dim: int = _EMBED_DIM,
        vocab_size: int = _VOCAB_SIZE,
        seed: int = 42,
    ) -> None:
        self._dim = dim
        self._vocab_size = vocab_size
        if _NUMPY_AVAILABLE:
            rng = np.random.default_rng(seed)
            self._proj: Any = (
                rng.standard_normal((vocab_size, dim)).astype(np.float32)
                / math.sqrt(dim)
            )
        else:
            self._proj = None
        self._idf: dict[int, float] = {}
        self._doc_count = 0
        self._df: dict[int, int] = {}

    def _token_to_idx(self, token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self._vocab_size

    def update_idf(self, texts: list[str]) -> None:
        """Update IDF weights from a batch of texts."""
        for text in texts:
            for token in set(_tokenize(text)):
                idx = self._token_to_idx(token)
                self._df[idx] = self._df.get(idx, 0) + 1
        self._doc_count += len(texts)
        self._idf = {
            idx: math.log((self._doc_count + 1) / (count + 1)) + 1
            for idx, count in self._df.items()
        }

    def _tfidf_sparse(self, text: str) -> dict[int, float]:
        tokens = _tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) or 1
        return {
            self._token_to_idx(tok): (cnt / total) * self._idf.get(
                self._token_to_idx(tok), 1.0
            )
            for tok, cnt in tf.items()
        }

    def embed(self, text: str) -> list[float]:
        sparse = self._tfidf_sparse(text)
        if not _NUMPY_AVAILABLE or self._proj is None:
            # Pure-Python fallback: scatter-accumulate into a fixed-size list
            result = [0.0] * self._dim
            total_sq = sum(v * v for v in sparse.values()) ** 0.5 or 1.0
            for idx, val in sparse.items():
                result[idx % self._dim] += val / total_sq
            return result

        vec = np.zeros(self._vocab_size, dtype=np.float32)
        for idx, val in sparse.items():
            vec[idx] = val
        projected: Any = vec @ self._proj  # shape (dim,)
        norm = float(np.linalg.norm(projected))
        if norm > 0:
            projected = projected / norm
        return projected.tolist()


class OpenAIEmbeddingEngine(EmbeddingEngine):
    """Embedding engine backed by an OpenAI-compatible /embeddings endpoint."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self._model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{self._base_url}/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return [item["embedding"] for item in data["data"]]


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _gpu_device() -> str:
    """
    Return the best available torch device string.

    Priority: ``cuda`` → ``mps`` (Apple Silicon) → ``cpu``.
    """
    if _TORCH_AVAILABLE:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


class LocalGPUEmbeddingEngine(EmbeddingEngine):
    """
    GPU-accelerated embedding engine using *sentence-transformers*.

    Uses ``cuda`` when available, falls back to ``mps`` (Apple Silicon) then
    ``cpu``.  Model files are downloaded from HuggingFace Hub on first use
    and cached at ``MSR_HF_CACHE_DIR`` (default: ``/tmp/hf_cache``).

    This engine is selected when ``MSR_USE_LOCAL_GPU=true`` and
    ``sentence-transformers`` is installed.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
        Default: ``sentence-transformers/all-MiniLM-L6-v2`` (384-dim, ~90 MB).
    device : str | None
        Torch device string (``"cuda"``, ``"mps"``, ``"cpu"``).
        Auto-detected if ``None``.
    batch_size : int
        Batch size passed to ``SentenceTransformer.encode()``.

    Raises
    ------
    ImportError
        If ``sentence-transformers`` is not installed.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "LocalGPUEmbeddingEngine requires 'sentence-transformers'. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._batch_size = batch_size
        self._device = device or _gpu_device()
        # Set HuggingFace cache to writable Lambda /tmp directory
        cache_dir = os.environ.get("MSR_HF_CACHE_DIR", "/tmp/hf_cache")
        os.environ.setdefault("HF_HOME", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        self._model = SentenceTransformer(model_name, device=self._device, cache_folder=cache_dir)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    @property
    def device(self) -> str:
        """Torch device being used (``"cuda"``, ``"mps"``, or ``"cpu"``)."""
        return self._device


class LocalGPULLM:
    """
    GPU-accelerated text-generation LLM using HuggingFace *transformers*.

    Provides a ``generate(messages, max_new_tokens)`` method that mirrors the
    interface of the OpenAI chat-completions endpoint, so it can be used as a
    drop-in replacement inside the RAG pipeline.

    Uses ``cuda`` when available, falls back to ``mps`` then ``cpu``.
    Model files are cached at ``MSR_HF_CACHE_DIR`` (default: ``/tmp/hf_cache``).

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
        Default: ``TinyLlama/TinyLlama-1.1B-Chat-v1.0`` (~600 MB on disk,
        fits in Lambda's 10 GB container limit).
    device : str | None
        Torch device string.  Auto-detected if ``None``.
    load_in_8bit : bool
        Quantise to 8-bit integers via *bitsandbytes* to halve GPU memory use.
        Requires ``bitsandbytes`` to be installed and a CUDA device.
    torch_dtype : str | None
        Floating-point dtype for model weights: ``"float16"`` (default on GPU)
        or ``"float32"`` (default on CPU).

    Raises
    ------
    ImportError
        If ``transformers`` is not installed.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str | None = None,
        load_in_8bit: bool = False,
        torch_dtype: str | None = None,
    ) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "LocalGPULLM requires 'transformers'. "
                "Install it with: pip install transformers accelerate"
            ) from exc

        self._device = device or _gpu_device()
        cache_dir = os.environ.get("MSR_HF_CACHE_DIR", "/tmp/hf_cache")
        os.environ.setdefault("HF_HOME", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        # Select dtype: float16 on GPU, float32 on CPU
        if torch_dtype is None:
            torch_dtype = "float16" if self._device != "cpu" and _TORCH_AVAILABLE else "float32"

        _torch_dtype = getattr(torch, torch_dtype) if _TORCH_AVAILABLE else None

        load_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "torch_dtype": _torch_dtype,
        }
        if load_in_8bit and self._device == "cuda":
            load_kwargs["load_in_8bit"] = True
        elif self._device != "cpu":
            load_kwargs["device_map"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        pipe_device: Any = -1  # CPU for transformers pipeline
        if self._device == "cuda":
            pipe_device = 0
        elif self._device == "mps":
            pipe_device = "mps"

        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=pipe_device if not load_kwargs.get("device_map") else None,
        )
        self._model_name = model_name

    def generate(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Generate a response for a chat-style message list.

        Applies the tokenizer's chat template when available (e.g. TinyLlama
        uses the ChatML format), otherwise concatenates role/content pairs.

        Parameters
        ----------
        messages : list[dict[str, str]]
            List of ``{"role": ..., "content": ...}`` dicts.
        max_new_tokens : int
            Maximum number of new tokens to generate.

        Returns
        -------
        str
            The assistant's reply text (stripped).
        """
        tokenizer = self._pipeline.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(
                f"<|{m['role']}|>\n{m['content']}" for m in messages
            ) + "\n<|assistant|>\n"

        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
            return_full_text=False,
        )
        return outputs[0]["generated_text"].strip()

    @property
    def device(self) -> str:
        """Torch device being used."""
        return self._device


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_idx: int
    embedding: list[float] = dataclasses.field(default_factory=list)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SourceInsight:
    """
    LLM-generated insights for a document source.

    Inspired by open-notebook's ``SourceInsight`` model and the
    ``source_transformation`` graph that automatically generates summaries,
    topics, and key facts when a source is ingested.
    """

    source: str
    summary: str
    topics: list[str]
    key_facts: list[str]


@dataclasses.dataclass
class SubQuery:
    """One search sub-query produced by the query-decomposition step."""

    term: str
    instructions: str


# ---------------------------------------------------------------------------
# Knowledge base (persistent)
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Persistent knowledge base supporting hybrid dense + sparse retrieval.

    Data is saved to *store_dir* as:

    - ``chunks.json``     – chunk text and metadata
    - ``embeddings.npy``  – dense embedding matrix (numpy, if available)
    - ``insights.json``   – LLM-generated source insights
    - ``tfidf.json``      – document-frequency counts for TF-IDF
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        store_dir: str | Path | None = None,
    ) -> None:
        self._engine = embedding_engine
        self._chunks: list[DocumentChunk] = []
        self._insights: dict[str, SourceInsight] = {}
        self._store_dir = Path(
            store_dir or os.environ.get("MSR_KB_DIR", "./kb_store")
        )
        # TF-IDF state for sparse-retrieval component of hybrid scoring
        self._df: dict[str, int] = {}
        self._doc_count = 0
        self._idf_weights: dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_document(
        self,
        text: str,
        source: str = "",
        insight: SourceInsight | None = None,
        *,
        chunk_words: int = _CHUNK_WORDS,
        chunk_overlap: int = _CHUNK_OVERLAP,
    ) -> int:
        """
        Chunk, embed, and store *text*.

        Returns the number of chunks added.
        """
        raw_chunks = _chunk_text(text, chunk_words, chunk_overlap)

        # Update local IDF for random-projection engine
        if isinstance(self._engine, RandomProjectionEmbeddingEngine):
            self._engine.update_idf(raw_chunks)

        embeddings = self._engine.embed_batch(raw_chunks)

        for idx, (chunk_text, emb) in enumerate(zip(raw_chunks, embeddings)):
            self._chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source=source,
                    chunk_idx=idx,
                    embedding=emb,
                )
            )

        if insight:
            self._insights[source] = insight

        self._update_tfidf_stats(raw_chunks)
        self._save()
        return len(raw_chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self, query: str, top_k: int = 5, alpha: float = 0.7
    ) -> list[dict[str, Any]]:
        """
        Hybrid retrieval combining dense vector similarity (*alpha*) and
        sparse TF-IDF similarity (1 − *alpha*).
        """
        if not self._chunks:
            return []

        q_emb = self._engine.embed(query)
        q_sparse = self._tfidf_vector(query)

        scores: list[tuple[float, int]] = []
        for i, chunk in enumerate(self._chunks):
            dense = _cosine_dense(q_emb, chunk.embedding)
            sparse = _cosine_sparse(q_sparse, self._tfidf_vector(chunk.text))
            scores.append((alpha * dense + (1 - alpha) * sparse, i))

        scores.sort(reverse=True)
        results: list[dict[str, Any]] = []
        for score, idx in scores[:top_k]:
            chunk = self._chunks[idx]
            entry: dict[str, Any] = {
                "text": chunk.text,
                "source": chunk.source,
                "score": round(score, 4),
                "chunk_idx": chunk.chunk_idx,
            }
            if chunk.source in self._insights:
                ins = self._insights[chunk.source]
                entry["source_summary"] = ins.summary
                entry["source_topics"] = ins.topics
            results.append(entry)

        return results

    def get_all_insights(self) -> list[SourceInsight]:
        return list(self._insights.values())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
            chunks_data = [
                {
                    "text": c.text,
                    "source": c.source,
                    "chunk_idx": c.chunk_idx,
                    "metadata": c.metadata,
                }
                for c in self._chunks
            ]
            (self._store_dir / "chunks.json").write_text(
                json.dumps(chunks_data, indent=2), encoding="utf-8"
            )
            if _NUMPY_AVAILABLE and self._chunks:
                embs = np.array(
                    [c.embedding for c in self._chunks], dtype=np.float32
                )
                np.save(str(self._store_dir / "embeddings.npy"), embs)
            insights_data = {
                src: dataclasses.asdict(ins)
                for src, ins in self._insights.items()
            }
            (self._store_dir / "insights.json").write_text(
                json.dumps(insights_data, indent=2), encoding="utf-8"
            )
            tfidf_data = {"df": self._df, "doc_count": self._doc_count}
            (self._store_dir / "tfidf.json").write_text(
                json.dumps(tfidf_data, indent=2), encoding="utf-8"
            )
        except OSError:
            pass  # Non-fatal persistence failure

    def _load(self) -> None:
        chunks_path = self._store_dir / "chunks.json"
        embeddings_path = self._store_dir / "embeddings.npy"
        insights_path = self._store_dir / "insights.json"
        tfidf_path = self._store_dir / "tfidf.json"

        if not chunks_path.exists():
            return
        try:
            chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
            embeddings: list[list[float]] = []
            if _NUMPY_AVAILABLE and embeddings_path.exists():
                embeddings = np.load(str(embeddings_path)).tolist()

            for i, c in enumerate(chunks_data):
                emb = embeddings[i] if i < len(embeddings) else []
                self._chunks.append(
                    DocumentChunk(
                        text=c["text"],
                        source=c["source"],
                        chunk_idx=c["chunk_idx"],
                        embedding=emb,
                        metadata=c.get("metadata", {}),
                    )
                )

            if insights_path.exists():
                raw = json.loads(insights_path.read_text(encoding="utf-8"))
                for src, data in raw.items():
                    self._insights[src] = SourceInsight(**data)

            if tfidf_path.exists():
                td = json.loads(tfidf_path.read_text(encoding="utf-8"))
                self._df = dict(td.get("df", {}))
                self._doc_count = td.get("doc_count", 0)
                self._idf_weights = {
                    token: math.log((self._doc_count + 1) / (cnt + 1)) + 1
                    for token, cnt in self._df.items()
                }
        except (OSError, KeyError, ValueError):
            # Corrupted store – start fresh
            self._chunks = []
            self._insights = {}

    # ------------------------------------------------------------------
    # TF-IDF helpers
    # ------------------------------------------------------------------

    def _update_tfidf_stats(self, texts: list[str]) -> None:
        for text in texts:
            for token in set(_tokenize(text)):
                self._df[token] = self._df.get(token, 0) + 1
        self._doc_count += len(texts)
        self._idf_weights = {
            token: math.log((self._doc_count + 1) / (cnt + 1)) + 1
            for token, cnt in self._df.items()
        }

    def _tfidf_vector(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) or 1
        return {
            token: (cnt / total) * self._idf_weights.get(token, 1.0)
            for token, cnt in tf.items()
        }


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _call_llm(
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int = 1024,
) -> str:
    """Call an OpenAI-compatible chat-completions endpoint."""
    payload = json.dumps(
        {"model": model, "messages": messages, "max_tokens": max_tokens}
    ).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Query decomposition (open-notebook ask.py pattern)
# ---------------------------------------------------------------------------

def _decompose_query(
    question: str,
    api_key: str,
    base_url: str,
    model: str,
    *,
    generate_fn: Callable[[list[dict[str, str]], int], str] | None = None,
) -> list[SubQuery]:
    """
    Use the LLM to decompose *question* into ≤ 5 targeted sub-queries.

    Inspired by open-notebook's ``call_model_with_messages`` in ``ask.py``
    which generates a ``Strategy`` containing multiple ``Search`` objects,
    each with a search term and extraction instructions.

    Falls back to a single sub-query equal to the original question if the
    LLM call fails or the response cannot be parsed.

    Parameters
    ----------
    generate_fn : callable, optional
        If provided, used for generation instead of the OpenAI API.
        Signature: ``generate_fn(messages, max_tokens) -> str``.
        When ``None`` the OpenAI API is used (``api_key`` must be set).
    """
    system = textwrap.dedent("""\
        You are a search strategy planner for a Molten Salt Reactor (MSR) \
knowledge base.
        Given a question, generate up to 5 targeted search queries that together \
cover all aspects needed to answer the question fully.

        For each search query provide:
        - "term": a short search phrase (2-8 words)
        - "instructions": what specific information to extract from the results

        Respond ONLY with valid JSON in this exact format:
        {
          "searches": [
            {"term": "...", "instructions": "..."}
          ]
        }
    """)

    def _do_generate(messages: list[dict[str, str]], max_tokens: int) -> str:
        if generate_fn is not None:
            return generate_fn(messages, max_tokens)
        return _call_llm(messages, api_key, base_url, model, max_tokens)

    try:
        response = _do_generate(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {question}"},
            ],
            512,
        )
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")
        data = json.loads(match.group())
        queries = [
            SubQuery(term=s["term"], instructions=s["instructions"])
            for s in data.get("searches", [])
            if s.get("term")
        ]
        return queries or [SubQuery(term=question, instructions="Extract all relevant information.")]
    except Exception as exc:  # noqa: BLE001
        import sys
        print(f"[RAG] Query decomposition failed ({exc!r}), falling back to single query.", file=sys.stderr)
        return [SubQuery(term=question, instructions="Extract all relevant information.")]


# ---------------------------------------------------------------------------
# Source insight extraction (open-notebook source-transformation pattern)
# ---------------------------------------------------------------------------

def _extract_insight(
    text: str,
    source_name: str,
    api_key: str,
    base_url: str,
    model: str,
    *,
    generate_fn: Callable[[list[dict[str, str]], int], str] | None = None,
) -> SourceInsight:
    """
    Generate a structured :class:`SourceInsight` for a document using the LLM.

    Inspired by open-notebook's source transformation/insight pipeline in
    ``source.py`` which generates insights (summaries, key points, etc.) for
    each ingested source and stores them alongside the document embeddings.

    Parameters
    ----------
    generate_fn : callable, optional
        If provided, used for generation instead of the OpenAI API.
        Signature: ``generate_fn(messages, max_tokens) -> str``.
    """
    preview = text[:3000]
    system = textwrap.dedent("""\
        You are an expert analyst of Molten Salt Reactor (MSR) technical documents.
        Analyse the given document excerpt and extract:
        1. A concise summary (2-3 sentences).
        2. Key technical topics covered.
        3. Important facts, measured values, or safety limits mentioned.

        Respond ONLY with valid JSON:
        {
          "summary": "...",
          "topics": ["topic1", "topic2"],
          "key_facts": ["fact1", "fact2"]
        }
    """)

    def _do_generate(messages: list[dict[str, str]], max_tokens: int) -> str:
        if generate_fn is not None:
            return generate_fn(messages, max_tokens)
        return _call_llm(messages, api_key, base_url, model, max_tokens)

    try:
        response = _do_generate(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Document:\n{preview}"},
            ],
            512,
        )
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")
        data = json.loads(match.group())
        return SourceInsight(
            source=source_name,
            summary=data.get("summary", ""),
            topics=data.get("topics", []),
            key_facts=data.get("key_facts", []),
        )
    except Exception as exc:  # noqa: BLE001
        import sys
        print(
            f"[RAG] Insight extraction failed for '{source_name}' ({exc!r}), "
            "storing empty insight.",
            file=sys.stderr,
        )
        return SourceInsight(source=source_name, summary="[Insight extraction failed]", topics=[], key_facts=[])


# ---------------------------------------------------------------------------
# Enhanced RAG pipeline
# ---------------------------------------------------------------------------

class MSRDigitalTwinRAG:
    """
    Multi-step RAG pipeline for the MSR data layer knowledge base.

    Enhancements over the original TF-IDF implementation, inspired by the
    open-notebook project:

    1. **Dense vector embeddings** – OpenAI API when available, deterministic
       random-projection fallback (numpy).
    2. **Source insights** – LLM-generated summary, topics, and key facts are
       stored alongside chunks and surfaced in search results.
    3. **Query decomposition** – the LLM breaks the question into ≤ 5
       targeted sub-queries with specific extraction instructions.
    4. **Parallel search** – sub-queries execute concurrently.
    5. **Sub-answer extraction** – the LLM distils each set of search results.
    6. **Final synthesis** – a second LLM pass combines sub-answers and live
       reactor data into a coherent, well-cited answer.
    7. **Hybrid retrieval** – dense cosine + sparse TF-IDF weighted average.
    8. **Persistent knowledge base** – avoids re-embedding on restart.
    9. **Local GPU models** – sentence-transformers for embeddings and a
       HuggingFace causal-LM for generation when ``MSR_USE_LOCAL_GPU=true``
       (CUDA/MPS when available, CPU fallback).
    """

    def __init__(self, docs_dir: str | Path | None = None) -> None:
        self._api_key = os.environ.get("MSR_OPENAI_API_KEY", "")
        self._base_url = os.environ.get(
            "MSR_OPENAI_BASE_URL", "https://api.openai.com/v1"
        )
        self._model = os.environ.get("MSR_OPENAI_MODEL", "gpt-4o-mini")
        self._embed_model = os.environ.get(
            "MSR_EMBED_MODEL", "text-embedding-3-small"
        )

        # Local GPU configuration
        _use_local_gpu = os.environ.get("MSR_USE_LOCAL_GPU", "").lower() in (
            "1", "true", "yes"
        )
        _local_embed_model = os.environ.get(
            "MSR_LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        _local_llm_model = os.environ.get(
            "MSR_LOCAL_LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )

        # Engine selection: local GPU > OpenAI API > random projection
        self._local_llm: LocalGPULLM | None = None
        if _use_local_gpu:
            self._embed_engine: EmbeddingEngine = LocalGPUEmbeddingEngine(
                model_name=_local_embed_model,
            )
            self._local_llm = LocalGPULLM(model_name=_local_llm_model)
            _device = self._embed_engine.device  # type: ignore[attr-defined]
            print(f"[RAG] Local GPU mode enabled (device={_device}).")
        elif self._api_key:
            self._embed_engine = OpenAIEmbeddingEngine(
                api_key=self._api_key,
                base_url=self._base_url,
                model=self._embed_model,
            )
        else:
            self._embed_engine = RandomProjectionEmbeddingEngine()

        self._kb = KnowledgeBase(self._embed_engine)

        # Load documents
        _docs = str(docs_dir or os.environ.get("MSR_DOCS_DIR", "./docs"))
        loaded = self._load_directory(_docs)
        if loaded:
            print(f"[RAG] Loaded {loaded} new document chunks from '{_docs}'.")
        else:
            print(
                f"[RAG] No new documents found in '{_docs}'. "
                "Using existing knowledge base."
                if self._kb._chunks
                else f"[RAG] No documents found in '{_docs}'. RAG context will be empty."
            )

    # ------------------------------------------------------------------
    # LLM dispatch
    # ------------------------------------------------------------------

    def _llm_generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
    ) -> str:
        """
        Route a generation request to the configured LLM backend.

        Priority:
        1. Local GPU LLM (when ``MSR_USE_LOCAL_GPU=true``)
        2. OpenAI-compatible API (when ``MSR_OPENAI_API_KEY`` is set)
        3. Raises ``RuntimeError`` if neither is available.
        """
        if self._local_llm is not None:
            return self._local_llm.generate(messages, max_new_tokens=max_tokens)
        if self._api_key:
            return _call_llm(messages, self._api_key, self._base_url, self._model, max_tokens)
        raise RuntimeError(
            "No LLM backend configured. "
            "Set MSR_OPENAI_API_KEY for the OpenAI API, or "
            "MSR_USE_LOCAL_GPU=true for local GPU inference."
        )

    def _has_llm(self) -> bool:
        """Return True if an LLM backend (local GPU or OpenAI API) is configured."""
        return self._local_llm is not None or bool(self._api_key)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add_document(self, text: str, source: str = "") -> int:
        """Programmatically add a document to the knowledge base."""
        insight = None
        if self._has_llm():
            insight = _extract_insight(
                text,
                source,
                self._api_key,
                self._base_url,
                self._model,
                generate_fn=self._llm_generate,
            )
        return self._kb.add_document(text, source=source, insight=insight)

    def load_msr_archive(self, max_docs: int = 0) -> int:
        """
        Ingest new OCR documents from ``pranavkantgaur/msr-archive``.

        Delegates to :class:`~msr_kb_sources.MSRArchiveLoader`.
        Documents already in the knowledge base are skipped.

        Parameters
        ----------
        max_docs:
            Maximum number of new files to ingest (``0`` = no limit).

        Returns
        -------
        int
            Number of documents newly added.
        """
        from msr_kb_sources import MSRArchiveLoader  # noqa: PLC0415
        loader = MSRArchiveLoader()
        return loader.ingest(self, max_docs=max_docs)

    def update_openalex(self, max_docs: int | None = None) -> int:
        """
        Ingest new papers from the OpenAlex API (MSR experimental data and
        TMSR-LF1 SINAP targeted queries).

        Delegates to :class:`~msr_kb_sources.OpenAlexLoader`.

        Parameters
        ----------
        max_docs:
            Maximum number of new papers to ingest (``None`` uses the
            ``MSR_OPENALEX_MAX_RESULTS`` env var, default 100).

        Returns
        -------
        int
            Number of documents newly added.
        """
        from msr_kb_sources import OpenAlexLoader  # noqa: PLC0415
        loader = OpenAlexLoader()
        return loader.ingest(self, max_docs=max_docs)

    def answer(self, question: str, top_k: int = 5) -> str:
        """
        Answer *question* using the multi-step RAG pipeline.

        If no LLM is configured (neither ``MSR_OPENAI_API_KEY`` nor
        ``MSR_USE_LOCAL_GPU=true``), returns a structured context summary
        that the caller can pass to their own LLM.
        """
        reactor_context = self._fetch_reactor_context()

        if not self._has_llm():
            chunks = self._kb.search(question, top_k=top_k)
            prompt = self._build_simple_prompt(
                question, self._format_chunks(chunks), reactor_context
            )
            return (
                "[No LLM configured – set MSR_OPENAI_API_KEY or "
                "MSR_USE_LOCAL_GPU=true to enable generation]\n\n"
                + prompt
            )

        # Step 1 – decompose question into sub-queries
        sub_queries = _decompose_query(
            question,
            api_key=self._api_key,
            base_url=self._base_url,
            model=self._model,
            generate_fn=self._llm_generate,
        )

        # Step 2 – parallel search + sub-answer extraction
        sub_answers: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = {
                pool.submit(self._run_sub_query, sq, top_k): sq
                for sq in sub_queries
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    sub_answers.append(result)

        # Step 3 – synthesize final answer
        return self._synthesize(question, sub_answers, reactor_context)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_directory(self, docs_dir: str | Path) -> int:
        docs_dir = Path(docs_dir)
        if not docs_dir.is_dir():
            return 0
        already_loaded = {c.source for c in self._kb._chunks}
        total = 0
        paths = sorted(
            set(list(docs_dir.rglob("*.md")) + list(docs_dir.rglob("*.txt")))
        )
        for path in paths:
            if str(path) in already_loaded:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                if not text.strip():
                    continue
                insight = None
                if self._has_llm():
                    insight = _extract_insight(
                        text,
                        str(path),
                        self._api_key,
                        self._base_url,
                        self._model,
                        generate_fn=self._llm_generate,
                    )
                total += self._kb.add_document(
                    text, source=str(path), insight=insight
                )
            except OSError:
                continue
        return total

    def _run_sub_query(self, sub_query: SubQuery, top_k: int) -> str:
        """Search the KB for one sub-query and return a focused partial answer."""
        chunks = self._kb.search(sub_query.term, top_k=top_k)
        if not chunks:
            return ""

        doc_context = self._format_chunks(chunks)

        # Collect distinct source insights referenced by results
        seen: set[str] = set()
        insight_lines: list[str] = []
        for c in chunks:
            src = c.get("source", "")
            if src and src not in seen and "source_summary" in c:
                seen.add(src)
                insight_lines.append(
                    f"[Source: {src}]\n"
                    f"Summary: {c['source_summary']}\n"
                    f"Topics: {', '.join(c.get('source_topics', []))}"
                )

        parts = [
            f"Search term: {sub_query.term}",
            f"Instructions: {sub_query.instructions}",
            "",
            "## Retrieved Documents",
            doc_context,
        ]
        if insight_lines:
            parts += ["", "## Source Insights", "\n".join(insight_lines)]
        parts += [
            "",
            "Extract the information relevant to the instructions above.",
        ]

        try:
            return self._llm_generate(
                [{"role": "user", "content": "\n".join(parts)}],
                512,
            )
        except Exception as exc:  # noqa: BLE001
            import sys
            print(f"[RAG] Sub-query '{sub_query.term}' LLM call failed ({exc!r}).", file=sys.stderr)
            return f"[Sub-query search error: {exc}]"

    def _synthesize(
        self,
        question: str,
        sub_answers: list[str],
        reactor_context: str,
    ) -> str:
        """Combine sub-answers + live reactor data into a final answer."""
        findings = "\n\n".join(
            f"[Finding {i + 1}]\n{ans}"
            for i, ans in enumerate(sub_answers)
            if ans
        ) or "[No relevant information found in the knowledge base]"

        # Include top-level source insights in synthesis prompt
        all_insights = self._kb.get_all_insights()
        insight_block = ""
        if all_insights:
            lines = ["Knowledge base covers these topics:"]
            for ins in all_insights[:5]:
                if ins.topics:
                    lines.append(f"  • {ins.source}: {', '.join(ins.topics[:4])}")
            insight_block = "\n## Knowledge Base Overview\n" + "\n".join(lines)

        prompt = textwrap.dedent(f"""\
            You are an expert assistant for Molten Salt Reactor (MSR) design,
            construction, and operations.
            Synthesise the following research findings and live plant data to
            answer the question comprehensively.

            ## Live Plant Data
            {reactor_context}
            {insight_block}

            ## Research Findings
            {findings}

            ## Question
            {question}

            ## Final Answer
            Provide a clear, well-organised answer. Reference specific values
            and safety limits where relevant.
        """)

        try:
            return self._llm_generate(
                [{"role": "user", "content": prompt}],
                1024,
            )
        except Exception as exc:  # noqa: BLE001
            import sys
            print(f"[RAG] Final synthesis LLM call failed ({exc!r}).", file=sys.stderr)
            return (
                f"[Final synthesis failed: {exc}. Returning raw research findings below:]\n\n"
                + findings
            )

    def _fetch_reactor_context(self) -> str:
        try:
            with MSRDigitalTwinClient() as client:
                status = client.get_reactor_status()
                alarms = client.get_active_alarms()
            lines = ["Current plant state:"]
            for key, value in status.items():
                lines.append(f"  {key}: {value}")
            if alarms.get("alarm_count", 0) > 0:
                lines.append(f"\nActive alarms ({alarms['alarm_count']}):")
                for alarm in alarms["alarms"]:
                    lines.append(
                        f"  [{alarm['severity']}] {alarm['alarm_id']}: "
                        f"{alarm['sensor']} = {alarm['value']}"
                    )
            else:
                lines.append("\nNo active alarms.")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            return f"[Could not fetch plant state: {exc}]"

    @staticmethod
    def _format_chunks(chunks: list[dict[str, Any]]) -> str:
        if not chunks:
            return "[No relevant documents found]"
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            text = textwrap.shorten(chunk["text"], width=800, placeholder="…")
            score = chunk.get("score", 0.0)
            parts.append(
                f"[Document {i} | source: {source} | score: {score:.4f}]\n{text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _build_simple_prompt(
        question: str, doc_context: str, reactor_context: str
    ) -> str:
        return textwrap.dedent(f"""\
            You are an expert assistant for Molten Salt Reactor (MSR) design,
            construction, and operations.
            Use the following live plant data and reference documents to answer the question.

            ## Live Plant Data
            {reactor_context}

            ## Reference Documents
            {doc_context}

            ## Question
            {question}

            ## Answer
        """)


# ---------------------------------------------------------------------------
# Compatibility shims (kept for any code that imported these from v1)
# ---------------------------------------------------------------------------

def json_encode(obj: Any) -> str:
    return json.dumps(obj)


def json_decode(s: str) -> Any:
    return json.loads(s)


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
    answer_text = rag.answer(question)
    print(answer_text)
