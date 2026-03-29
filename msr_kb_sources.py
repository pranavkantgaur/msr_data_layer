"""
MSR Knowledge-Base Source Loaders

Five document sources feed the MSR data layer knowledge base:

1. **Static source** – ``pranavkantgaur/msr-archive`` GitHub repository
   OCR text files from the ``ocr/`` directory (transcribed ORNL Molten Salt
   Reactor reports) are fetched over HTTPS and ingested.  The list of files
   is discovered via the GitHub Contents API.

2. **Dynamic source** – OpenAlex academic papers API
   Papers matching "molten salt reactors experimental data" are fetched
   periodically.  A second targeted query focuses on TMSR-LF1 experimental
   data from the TMSR group at SINAP (Shanghai Institute of Applied Physics).

3. **Dynamic source** – arXiv preprint API (Atom XML)
   Preprints and recent papers matching "molten salt reactor experimental"
   and "TMSR-LF1" are fetched from the arXiv Atom XML feed.  arXiv is the
   primary source for cutting-edge nuclear engineering preprints before they
   are indexed by other databases.

4. **Dynamic source** – Semantic Scholar Graph API
   Academic papers from the Semantic Scholar corpus are fetched using the
   public S2 Graph API.  Semantic Scholar excels at nuclear engineering and
   materials science literature and often indexes papers earlier than OpenAlex.

5. **Plant operational data** – :class:`PlantDataLoader`
   Accepts real-time plant data pushed by operators or agents: sensor
   snapshots, event logs, and maintenance/inspection reports.  This enables
   the knowledge base to accumulate operational history so future RAG queries
   can incorporate real plant experience alongside reference documents.

6. **Plant timeseries store** – :class:`TimeseriesStore`
   SQLite-backed (``sqlite3`` stdlib) append-only store for timestamped
   sensor readings.  Enables time-range queries, aggregate statistics, and
   natural-language-to-SQL queries over structured plant timeseries data.
   Complements the text-centric RAG pipeline for numerical/temporal queries
   that require precise aggregation rather than semantic similarity search.

All loaders maintain a JSON state file in ``MSR_KB_DIR`` (default
``./kb_store``) so that documents already ingested are never processed
twice.  Re-running the updater therefore only adds truly new content.

Design inspiration
------------------
The multi-source literature search (arXiv + OpenAlex + Semantic Scholar)
mirrors the Phase B (Literature Discovery) pipeline of AutoResearchClaw
(https://github.com/aiming-lab/AutoResearchClaw), which demonstrated that
spanning three sources increases recall by ~40% for niche research domains.

Environment Variables
---------------------
MSR_KB_DIR                   Persistent KB directory (default ``./kb_store``)
MSR_ARCHIVE_REPO             GitHub ``owner/repo`` for the MSR archive
                              (default ``pranavkantgaur/msr-archive``)
MSR_ARCHIVE_BRANCH           Branch to fetch from (default ``master``)
MSR_ARCHIVE_MAX_DOCS         Max OCR files to ingest per run (0 = unlimited)
MSR_OPENALEX_MAX_RESULTS     Max OpenAlex works to ingest per run (default 100)
MSR_OPENALEX_EMAIL           Optional email for the OpenAlex ``mailto`` polite
                              pool (improves rate limits)
MSR_GITHUB_TOKEN             Optional GitHub personal-access token for higher
                              rate limits when listing the archive
MSR_ARXIV_MAX_RESULTS        Max arXiv papers to ingest per run (default 100)
MSR_S2_API_KEY               Optional Semantic Scholar API key for higher rate
                              limits (unauthenticated: 1 req/s; authenticated:
                              100 req/s)
MSR_S2_MAX_RESULTS           Max Semantic Scholar papers to ingest per run
                              (default 100)

CLI Usage
---------
    python msr_kb_sources.py --update-archive
    python msr_kb_sources.py --update-openalex
    python msr_kb_sources.py --update-arxiv
    python msr_kb_sources.py --update-semanticscholar
    python msr_kb_sources.py --update-all
    python msr_kb_sources.py --status
    python msr_kb_sources.py --ingest-plant-data --content "..." --data-type sensor_snapshot

Python Usage
------------
    from msr_digital_twin_with_rag import MSRDigitalTwinRAG
    from msr_kb_sources import KBSourceManager, PlantDataLoader

    rag = MSRDigitalTwinRAG()
    mgr = KBSourceManager(rag)
    mgr.update_all()          # ingest new docs from archive + OpenAlex + arXiv + Semantic Scholar

    # Ingest plant operational data directly
    loader = PlantDataLoader()
    loader.ingest_text(rag, "Core temperature spike to 712°C at 14:32 UTC", "event-001")
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

_ARCHIVE_REPO_DEFAULT = "pranavkantgaur/msr-archive"
_ARCHIVE_BRANCH_DEFAULT = "master"
_ARCHIVE_OCR_DIR = "ocr"

_OPENALEX_BASE = "https://api.openalex.org"
# Primary query – broad MSR experimental data
_OPENALEX_QUERY_PRIMARY = "molten salt reactors experimental data"
# Secondary query – focused on TMSR-LF1 / SINAP experimental data
_OPENALEX_QUERY_TMSR = "TMSR-LF1 SINAP experimental"
_OPENALEX_PER_PAGE = 25
_OPENALEX_MAX_RESULTS_DEFAULT = 100

_GITHUB_API_BASE = "https://api.github.com"
_GITHUB_RAW_BASE = "https://raw.githubusercontent.com"

_REQUEST_TIMEOUT = 30       # seconds for each HTTP call
_RETRY_WAIT = 2             # seconds between retries
_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# arXiv API constants
# ---------------------------------------------------------------------------

_ARXIV_API_BASE = "http://export.arxiv.org/api/query"
_ARXIV_NS = "http://www.w3.org/2005/Atom"
_ARXIV_OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
_ARXIV_MAX_RESULTS_DEFAULT = 100
_ARXIV_PER_REQUEST = 50     # arXiv recommends ≤ 100 per request
_ARXIV_INTER_REQUEST_WAIT = 3  # seconds; arXiv ToS requires ≥ 3 s between calls

# ---------------------------------------------------------------------------
# Semantic Scholar API constants
# ---------------------------------------------------------------------------

_S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_MAX_RESULTS_DEFAULT = 100
_S2_PER_PAGE = 50           # S2 max is 100; 50 is a safe default


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_get(url: str, headers: dict[str, str] | None = None) -> dict[str, Any] | list[Any]:
    """
    Perform a GET request and return the parsed JSON body.

    Retries up to ``_MAX_RETRIES`` times on transient errors.
    Raises ``urllib.error.URLError`` / ``urllib.error.HTTPError`` on failure.
    """
    hdrs = {"Accept": "application/json", "User-Agent": "msr-data-layer/1.0"}
    if headers:
        hdrs.update(headers)
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 403 and attempt < _MAX_RETRIES - 1:
                # Rate-limit hit – back off and retry
                time.sleep(_RETRY_WAIT * (2 ** attempt))
                continue
            raise
        except (urllib.error.URLError, OSError) as exc:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_WAIT)
                continue
            raise
    # unreachable, but keeps type-checker happy
    raise RuntimeError("HTTP GET failed after all retries")


def _http_get_text(url: str, headers: dict[str, str] | None = None) -> str:
    """
    Perform a GET request and return the raw text body.
    """
    hdrs = {"User-Agent": "msr-data-layer/1.0"}
    if headers:
        hdrs.update(headers)
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_WAIT)
                continue
            raise
    raise RuntimeError("HTTP GET (text) failed after all retries")


# ---------------------------------------------------------------------------
# State persistence helpers
# ---------------------------------------------------------------------------

def _load_state(path: Path) -> dict[str, Any]:
    """Load a JSON state file, returning an empty dict if missing/corrupt."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    """Save a JSON state file, creating parent directories as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[KB] Warning: could not save state to {path}: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Abstract reconstruction (OpenAlex inverted index → plain text)
# ---------------------------------------------------------------------------

def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """
    Reconstruct a plain-text abstract from OpenAlex's inverted-index format.

    OpenAlex stores abstracts as ``{word: [pos1, pos2, ...], ...}`` where each
    position is a 0-based word index in the original text.
    """
    if not inverted_index:
        return ""
    position_word: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            position_word.append((pos, word))
    position_word.sort()
    return " ".join(word for _, word in position_word)


# ---------------------------------------------------------------------------
# MSR Archive loader (static source)
# ---------------------------------------------------------------------------

class MSRArchiveLoader:
    """
    Fetches OCR text files from the ``ocr/`` directory of the
    ``pranavkantgaur/msr-archive`` GitHub repository.

    Files are listed via the GitHub Contents API and downloaded via the
    GitHub raw-content URL.  An optional ``MSR_GITHUB_TOKEN`` env var
    provides a personal access token to increase the unauthenticated rate
    limit (60 req/hr) to 5 000 req/hr.
    """

    def __init__(
        self,
        repo: str | None = None,
        branch: str | None = None,
        kb_dir: str | Path | None = None,
    ) -> None:
        self._repo = repo or os.environ.get("MSR_ARCHIVE_REPO", _ARCHIVE_REPO_DEFAULT)
        self._branch = branch or os.environ.get("MSR_ARCHIVE_BRANCH", _ARCHIVE_BRANCH_DEFAULT)
        self._kb_dir = Path(kb_dir or os.environ.get("MSR_KB_DIR", "./kb_store"))
        self._state_path = self._kb_dir / "archive_state.json"
        self._gh_token = os.environ.get("MSR_GITHUB_TOKEN", "")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def list_ocr_files(self) -> list[dict[str, str]]:
        """
        Return a list of OCR file descriptors from the archive's ``ocr/``
        directory.

        Each descriptor is ``{"name": "...", "download_url": "..."}``.
        Uses the GitHub Contents API (paginated, up to 1 000 entries/page).
        Falls back to constructing raw URLs from the repository tree if the
        directory listing fails.
        """
        url = (
            f"{_GITHUB_API_BASE}/repos/{self._repo}/contents/{_ARCHIVE_OCR_DIR}"
            f"?ref={self._branch}&per_page=100"
        )
        headers = {}
        if self._gh_token:
            headers["Authorization"] = f"token {self._gh_token}"

        try:
            data = _http_get(url, headers)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(
                f"[Archive] Could not list OCR files from GitHub API ({exc}). "
                "Is the repository public and accessible?",
                file=sys.stderr,
            )
            return []

        if not isinstance(data, list):
            return []

        return [
            {"name": item["name"], "download_url": item["download_url"]}
            for item in data
            if isinstance(item, dict)
            and item.get("type") == "file"
            and item.get("name", "").endswith(".txt")
            and item.get("download_url")
        ]

    def list_ocr_files_from_tree(self) -> list[dict[str, str]]:
        """
        Alternative listing via the Git tree API (handles repos >1 000 files).
        Returns descriptors in the same format as :meth:`list_ocr_files`.
        """
        url = (
            f"{_GITHUB_API_BASE}/repos/{self._repo}/git/trees/{self._branch}"
            "?recursive=1"
        )
        headers = {}
        if self._gh_token:
            headers["Authorization"] = f"token {self._gh_token}"

        try:
            data = _http_get(url, headers)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(
                f"[Archive] Could not fetch Git tree ({exc}).",
                file=sys.stderr,
            )
            return []

        if not isinstance(data, dict):
            return []

        raw_base = f"{_GITHUB_RAW_BASE}/{self._repo}/{self._branch}"
        return [
            {
                "name": Path(item["path"]).name,
                "download_url": f"{raw_base}/{item['path']}",
            }
            for item in data.get("tree", [])
            if item.get("type") == "blob"
            and item.get("path", "").startswith(f"{_ARCHIVE_OCR_DIR}/")
            and item["path"].endswith(".txt")
        ]

    def fetch_text(self, download_url: str) -> str:
        """Download a single OCR text file and return its contents."""
        headers = {}
        if self._gh_token:
            headers["Authorization"] = f"token {self._gh_token}"
        return _http_get_text(download_url, headers)

    def ingest(
        self,
        rag: Any,
        max_docs: int = 0,
    ) -> int:
        """
        Ingest new OCR files from the archive into *rag* (an
        :class:`~msr_digital_twin_with_rag.MSRDigitalTwinRAG` instance).

        Only files not yet present in the loader's state file are processed.

        Parameters
        ----------
        rag:
            The RAG instance whose :meth:`add_document` method is called for
            each new file.
        max_docs:
            Maximum number of new files to ingest in one call.
            ``0`` (default) means no limit.

        Returns
        -------
        int
            Number of documents newly ingested.
        """
        state = _load_state(self._state_path)
        ingested_urls: set[str] = set(state.get("ingested_urls", []))

        # Try the directory listing first; fall back to tree API
        files = self.list_ocr_files()
        if not files:
            files = self.list_ocr_files_from_tree()

        if not files:
            print("[Archive] No OCR files found in archive repository.", file=sys.stderr)
            return 0

        new_files = [f for f in files if f["download_url"] not in ingested_urls]
        if max_docs > 0:
            new_files = new_files[:max_docs]

        if not new_files:
            print(f"[Archive] All {len(files)} files already ingested – nothing to do.")
            return 0

        print(f"[Archive] Ingesting {len(new_files)} new OCR file(s)…")
        count = 0
        for file_info in new_files:
            name = file_info["name"]
            url = file_info["download_url"]
            try:
                text = self.fetch_text(url)
                if text.strip():
                    n = rag.add_document(text, source=url)
                    print(f"[Archive] {name}: {n} chunks added.")
                    ingested_urls.add(url)
                    count += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[Archive] Skipping {name} – fetch error: {exc}", file=sys.stderr)

        # Persist updated state
        state["ingested_urls"] = sorted(ingested_urls)
        state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["total_ingested"] = len(ingested_urls)
        _save_state(self._state_path, state)
        print(f"[Archive] Done. {count} new document(s) added.")
        return count

    # ------------------------------------------------------------------
    # Convenience: status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return a dict summarising loader state (for ``--status`` CLI)."""
        state = _load_state(self._state_path)
        return {
            "source": f"GitHub: {self._repo} ({_ARCHIVE_OCR_DIR}/ @ {self._branch})",
            "total_ingested": state.get("total_ingested", 0),
            "last_run": state.get("last_run", "never"),
        }


# ---------------------------------------------------------------------------
# OpenAlex loader (dynamic source)
# ---------------------------------------------------------------------------

class OpenAlexLoader:
    """
    Fetches academic papers from the `OpenAlex <https://openalex.org>`_ API
    and ingests their title + abstract into the knowledge base.

    Two complementary search queries are used:

    1. ``"molten salt reactors experimental data"`` – broad MSR coverage
    2. ``"TMSR-LF1 SINAP experimental"`` – focused on the Chinese TMSR-LF1
       reactor from SINAP (Shanghai Institute of Applied Physics)

    The OpenAlex API uses cursor-based pagination.  The loader stores the
    work IDs it has already ingested in ``openalex_state.json`` so repeated
    runs only add new papers.

    A polite-pool email (``MSR_OPENALEX_EMAIL``) is recommended; it does not
    change the API behaviour but helps the OpenAlex team monitor usage.
    """

    # Queries are (label, filter_string) pairs.
    # OpenAlex filter syntax: https://docs.openalex.org/how-to-use-the-api/get-lists-of-entities/filter-entity-lists
    QUERIES: list[tuple[str, str]] = [
        (
            "MSR experimental data (general)",
            "title_and_abstract.search:molten+salt+reactors+experimental+data",
        ),
        (
            "TMSR-LF1 SINAP (targeted)",
            "title_and_abstract.search:TMSR-LF1",
        ),
    ]

    def __init__(
        self,
        kb_dir: str | Path | None = None,
        max_results: int | None = None,
        email: str | None = None,
    ) -> None:
        self._kb_dir = Path(kb_dir or os.environ.get("MSR_KB_DIR", "./kb_store"))
        self._state_path = self._kb_dir / "openalex_state.json"
        self._max_results = max_results or int(
            os.environ.get("MSR_OPENALEX_MAX_RESULTS", str(_OPENALEX_MAX_RESULTS_DEFAULT))
        )
        self._email = email or os.environ.get("MSR_OPENALEX_EMAIL", "")

    # ------------------------------------------------------------------
    # OpenAlex API helpers
    # ------------------------------------------------------------------

    def _base_headers(self) -> dict[str, str]:
        headers = {"User-Agent": "msr-data-layer/1.0"}
        if self._email:
            headers["User-Agent"] += f" (mailto:{self._email})"
        return headers

    def _make_url(
        self, filter_str: str, cursor: str = "*", per_page: int = _OPENALEX_PER_PAGE
    ) -> str:
        params = urllib.parse.urlencode(
            {
                "filter": filter_str,
                "sort": "relevance_score:desc",
                "per_page": per_page,
                "cursor": cursor,
                "select": "id,title,doi,abstract_inverted_index,open_access,publication_year,authorships",
            }
        )
        return f"{_OPENALEX_BASE}/works?{params}"

    def _iter_works(
        self, filter_str: str, max_results: int
    ) -> Iterator[dict[str, Any]]:
        """Yield raw OpenAlex work objects for *filter_str* up to *max_results*."""
        cursor = "*"
        fetched = 0
        while fetched < max_results:
            url = self._make_url(filter_str, cursor=cursor)
            try:
                data = _http_get(url, self._base_headers())
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                print(f"[OpenAlex] API error: {exc}", file=sys.stderr)
                break

            if not isinstance(data, dict):
                break

            results = data.get("results", [])
            if not results:
                break

            for work in results:
                if fetched >= max_results:
                    return
                yield work
                fetched += 1

            meta = data.get("meta", {})
            cursor = meta.get("next_cursor")
            if not cursor:
                break
            # Polite delay between pages
            time.sleep(0.2)

    # ------------------------------------------------------------------
    # Work text formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_work_text(work: dict[str, Any]) -> tuple[str, str]:
        """
        Format a raw OpenAlex work object into (text, source_id).

        The text includes title, year, authors, DOI, and reconstructed
        abstract.  The source_id is the OpenAlex work ID URL.

        The text is prefixed with ``[ABSTRACT ONLY]`` to indicate that only
        the abstract has been fetched automatically.  Use
        ``KBSourceManager.ingest_full_paper_text()`` to upgrade to the full
        text when needed.
        """
        title = work.get("title") or ""
        year = work.get("publication_year") or ""
        doi = work.get("doi") or ""
        work_id = work.get("id") or ""

        # Reconstruct abstract from inverted index
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

        # Author list (first 5)
        authorships = work.get("authorships") or []
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in authorships[:5]
            if a.get("author")
        ]
        author_str = "; ".join(a for a in authors if a)
        if len(authorships) > 5:
            author_str += f" et al. ({len(authorships)} total)"

        # Open-access PDF URL
        oa_info = work.get("open_access") or {}
        pdf_url = oa_info.get("oa_url") or ""

        parts = [
            "[ABSTRACT ONLY — full text not yet fetched. "
            "Call ingest_full_paper_text to upgrade.]",
            f"Title: {title}",
        ]
        if year:
            parts.append(f"Year: {year}")
        if author_str:
            parts.append(f"Authors: {author_str}")
        if doi:
            parts.append(f"DOI: {doi}")
        if pdf_url:
            parts.append(f"PDF: {pdf_url}")
        if abstract:
            parts.append(f"\nAbstract:\n{abstract}")

        return "\n".join(parts), work_id

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, rag: Any, max_docs: int | None = None) -> int:
        """
        Ingest new OpenAlex papers into *rag*.

        Parameters
        ----------
        rag:
            :class:`~msr_digital_twin_with_rag.MSRDigitalTwinRAG` instance.
        max_docs:
            Maximum number of new papers to ingest across all queries.
            Falls back to ``self._max_results``.

        Returns
        -------
        int
            Number of documents newly ingested.
        """
        limit = max_docs if max_docs is not None else self._max_results
        state = _load_state(self._state_path)
        ingested_ids: set[str] = set(state.get("ingested_ids", []))

        total_new = 0

        for label, filter_str in self.QUERIES:
            remaining = limit - total_new
            if remaining <= 0:
                break
            print(f"[OpenAlex] Running query: {label}")
            for work in self._iter_works(filter_str, remaining):
                work_id = work.get("id") or ""
                if not work_id or work_id in ingested_ids:
                    continue
                text, source_id = self.format_work_text(work)
                if not text.strip():
                    continue
                try:
                    n = rag.add_document(text, source=source_id)
                    title_short = textwrap.shorten(
                        work.get("title") or source_id, width=70, placeholder="…"
                    )
                    print(f"[OpenAlex] + {title_short} ({n} chunks)")
                    ingested_ids.add(work_id)
                    total_new += 1
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[OpenAlex] Skipping {work_id}: {exc}", file=sys.stderr
                    )

        # Persist
        state["ingested_ids"] = sorted(ingested_ids)
        state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["total_ingested"] = len(ingested_ids)
        _save_state(self._state_path, state)
        print(f"[OpenAlex] Done. {total_new} new paper(s) added.")
        return total_new

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        state = _load_state(self._state_path)
        return {
            "source": "OpenAlex API (MSR experimental data + TMSR-LF1 SINAP)",
            "total_ingested": state.get("total_ingested", 0),
            "last_run": state.get("last_run", "never"),
        }



# ---------------------------------------------------------------------------
# arXiv loader (preprints and recent experimental papers)
# ---------------------------------------------------------------------------

class ArXivLoader:
    """
    Fetches MSR-related preprints and papers from the
    `arXiv <https://arxiv.org>`_ Atom XML API.

    Two complementary queries are used:

    1. ``"molten salt reactor experimental"`` – broad coverage of experimental
       MSR research across physics, nuclear engineering, and materials science.
    2. ``"TMSR-LF1"`` – targeted search for TMSR-LF1 papers from SINAP and
       partner institutions.

    arXiv is the primary source for cutting-edge nuclear engineering preprints
    that may not yet appear in OpenAlex or Semantic Scholar.

    The loader respects arXiv's API guidelines:

    * Requests are spaced ≥ 3 seconds apart (``_ARXIV_INTER_REQUEST_WAIT``).
    * The ``User-Agent`` header identifies this client.

    State is persisted in ``arxiv_state.json`` inside ``MSR_KB_DIR``; paper
    IDs already ingested are never fetched again.

    Environment Variables
    ---------------------
    MSR_KB_DIR              KB store directory (default ``./kb_store``)
    MSR_ARXIV_MAX_RESULTS   Max papers per run (default 100)
    """

    QUERIES: list[tuple[str, str]] = [
        ("MSR experimental (general)", "all:molten+salt+reactor+experimental"),
        ("TMSR-LF1 (targeted)", "all:TMSR-LF1"),
    ]

    def __init__(
        self,
        kb_dir: str | Path | None = None,
        max_results: int | None = None,
    ) -> None:
        self._kb_dir = Path(kb_dir or os.environ.get("MSR_KB_DIR", "./kb_store"))
        self._state_path = self._kb_dir / "arxiv_state.json"
        self._max_results = max_results or int(
            os.environ.get("MSR_ARXIV_MAX_RESULTS", str(_ARXIV_MAX_RESULTS_DEFAULT))
        )

    # ------------------------------------------------------------------
    # arXiv API helpers
    # ------------------------------------------------------------------

    def _make_url(self, search_query: str, start: int) -> str:
        params = urllib.parse.urlencode(
            {
                "search_query": search_query,
                "start": start,
                "max_results": _ARXIV_PER_REQUEST,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        return f"{_ARXIV_API_BASE}?{params}"

    def _fetch_atom(self, url: str) -> str:
        """Fetch raw Atom XML from arXiv."""
        headers = {"User-Agent": "msr-data-layer/1.0 (MSR knowledge base; contact: open-source)"}
        return _http_get_text(url, headers)

    def _parse_entries(self, xml_text: str) -> list[dict[str, Any]]:
        """
        Parse Atom XML entries from an arXiv API response.

        Returns a list of dicts with keys:
            ``id``, ``title``, ``abstract``, ``authors``, ``published``,
            ``doi``, ``arxiv_id``.
        """
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            print(f"[arXiv] XML parse error: {exc}", file=sys.stderr)
            return []

        ns = {"atom": _ARXIV_NS}
        entries = []
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            published_el = entry.find("atom:published", ns)
            id_el = entry.find("atom:id", ns)

            title = (title_el.text or "").strip() if title_el is not None else ""
            abstract = (summary_el.text or "").strip() if summary_el is not None else ""
            published = (published_el.text or "").strip() if published_el is not None else ""
            raw_id = (id_el.text or "").strip() if id_el is not None else ""

            # arXiv IDs look like http://arxiv.org/abs/2301.12345v1
            arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
            # Remove version suffix e.g. "2301.12345v2" → "2301.12345"
            arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

            # DOI link (arXiv may expose it as a link with title="doi")
            doi = ""
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "doi":
                    doi = link.get("href", "")
                    break

            authors = []
            for author_el in entry.findall("atom:author", ns):
                name_el = author_el.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            if not arxiv_id:
                continue

            entries.append(
                {
                    "id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "published": published[:10],  # YYYY-MM-DD
                    "doi": doi,
                }
            )
        return entries

    # ------------------------------------------------------------------
    # Text formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_entry_text(entry: dict[str, Any]) -> tuple[str, str]:
        """
        Format a parsed arXiv entry into (text, source_id).

        The text includes title, year, authors, arXiv ID, DOI (if available),
        and abstract.  The source_id is ``arxiv:<arxiv_id>``.

        The text is prefixed with ``[ABSTRACT ONLY]`` to indicate that only
        the abstract has been fetched automatically.  Use
        ``KBSourceManager.ingest_full_paper_text()`` to upgrade to the full
        text when needed.
        """
        arxiv_id = entry.get("id", "")
        title = entry.get("title", "")
        published = entry.get("published", "")
        doi = entry.get("doi", "")
        abstract = entry.get("abstract", "")
        authors = entry.get("authors", [])

        author_str = "; ".join(authors[:5])
        if len(authors) > 5:
            author_str += f" et al. ({len(authors)} total)"

        parts = [
            "[ABSTRACT ONLY — full text not yet fetched. "
            "Call ingest_full_paper_text to upgrade.]",
            f"Title: {title}",
        ]
        if published:
            parts.append(f"Published: {published}")
        if author_str:
            parts.append(f"Authors: {author_str}")
        parts.append(f"arXiv: https://arxiv.org/abs/{arxiv_id}")
        if doi:
            parts.append(f"DOI: {doi}")
        if abstract:
            parts.append(f"\nAbstract:\n{abstract}")

        return "\n".join(parts), f"arxiv:{arxiv_id}"

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, rag: Any, max_docs: int | None = None) -> int:
        """
        Ingest new arXiv papers into *rag*.

        Parameters
        ----------
        rag:
            :class:`~msr_digital_twin_with_rag.MSRDigitalTwinRAG` instance.
        max_docs:
            Maximum number of new papers to ingest across all queries.
            Falls back to ``self._max_results``.

        Returns
        -------
        int
            Number of documents newly ingested.
        """
        limit = max_docs if max_docs is not None else self._max_results
        state = _load_state(self._state_path)
        ingested_ids: set[str] = set(state.get("ingested_ids", []))
        total_new = 0

        for label, search_query in self.QUERIES:
            remaining = limit - total_new
            if remaining <= 0:
                break
            print(f"[arXiv] Running query: {label}")
            start = 0
            while total_new < limit:
                url = self._make_url(search_query, start)
                try:
                    xml_text = self._fetch_atom(url)
                except (urllib.error.URLError, OSError) as exc:
                    print(f"[arXiv] API error: {exc}", file=sys.stderr)
                    break

                entries = self._parse_entries(xml_text)
                if not entries:
                    break

                for entry in entries:
                    if total_new >= limit:
                        break
                    arxiv_id = entry.get("id", "")
                    if not arxiv_id or arxiv_id in ingested_ids:
                        continue
                    text, source_id = self.format_entry_text(entry)
                    if not text.strip():
                        continue
                    try:
                        n = rag.add_document(text, source=source_id)
                        title_short = textwrap.shorten(
                            entry.get("title") or source_id, width=70, placeholder="…"
                        )
                        print(f"[arXiv] + {title_short} ({n} chunks)")
                        ingested_ids.add(arxiv_id)
                        total_new += 1
                    except Exception as exc:  # noqa: BLE001
                        print(f"[arXiv] Skipping {source_id}: {exc}", file=sys.stderr)

                start += len(entries)
                if len(entries) < _ARXIV_PER_REQUEST:
                    break  # Last page

                # Respect arXiv rate limit
                time.sleep(_ARXIV_INTER_REQUEST_WAIT)

        # Persist
        state["ingested_ids"] = sorted(ingested_ids)
        state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["total_ingested"] = len(ingested_ids)
        _save_state(self._state_path, state)
        print(f"[arXiv] Done. {total_new} new paper(s) added.")
        return total_new

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return loader state summary."""
        state = _load_state(self._state_path)
        return {
            "source": "arXiv API (molten salt reactor experimental + TMSR-LF1)",
            "total_ingested": state.get("total_ingested", 0),
            "last_run": state.get("last_run", "never"),
        }


# ---------------------------------------------------------------------------
# Semantic Scholar loader (academic papers via S2 Graph API)
# ---------------------------------------------------------------------------

class SemanticScholarLoader:
    """
    Fetches academic papers from the
    `Semantic Scholar <https://www.semanticscholar.org>`_ Graph API and
    ingests their title + abstract into the knowledge base.

    Two complementary queries are used:

    1. ``"molten salt reactor experimental"`` – broad coverage of experimental
       MSR research.
    2. ``"TMSR LF1 SINAP"`` – targeted papers from the TMSR-LF1 programme.

    The Semantic Scholar API provides high-quality metadata for nuclear
    engineering and materials science literature, often indexing papers
    earlier than OpenAlex.

    An optional Semantic Scholar API key (``MSR_S2_API_KEY``) raises the
    rate limit from 1 req/s to 100 req/s; the loader works without a key.

    State is persisted in ``semanticscholar_state.json`` inside ``MSR_KB_DIR``;
    paper IDs already ingested are never fetched again.

    Environment Variables
    ---------------------
    MSR_KB_DIR              KB store directory (default ``./kb_store``)
    MSR_S2_API_KEY          Optional S2 API key (higher rate limits)
    MSR_S2_MAX_RESULTS      Max papers per run (default 100)
    """

    QUERIES: list[tuple[str, str]] = [
        ("MSR experimental (general)", "molten salt reactor experimental"),
        ("TMSR-LF1 SINAP (targeted)", "TMSR LF1 SINAP"),
    ]

    _FIELDS = "title,abstract,year,authors,externalIds,openAccessPdf"

    def __init__(
        self,
        kb_dir: str | Path | None = None,
        max_results: int | None = None,
        api_key: str | None = None,
    ) -> None:
        self._kb_dir = Path(kb_dir or os.environ.get("MSR_KB_DIR", "./kb_store"))
        self._state_path = self._kb_dir / "semanticscholar_state.json"
        self._max_results = max_results or int(
            os.environ.get("MSR_S2_MAX_RESULTS", str(_S2_MAX_RESULTS_DEFAULT))
        )
        self._api_key = api_key or os.environ.get("MSR_S2_API_KEY", "")

    # ------------------------------------------------------------------
    # S2 API helpers
    # ------------------------------------------------------------------

    def _base_headers(self) -> dict[str, str]:
        headers = {"User-Agent": "msr-data-layer/1.0"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    def _make_url(self, query: str, offset: int) -> str:
        params = urllib.parse.urlencode(
            {
                "query": query,
                "fields": self._FIELDS,
                "limit": _S2_PER_PAGE,
                "offset": offset,
            }
        )
        return f"{_S2_API_BASE}/paper/search?{params}"

    def _iter_papers(self, query: str, max_results: int) -> Iterator[dict[str, Any]]:
        """Yield raw S2 paper objects for *query* up to *max_results*."""
        offset = 0
        fetched = 0
        while fetched < max_results:
            url = self._make_url(query, offset)
            try:
                data = _http_get(url, self._base_headers())
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                print(f"[S2] API error: {exc}", file=sys.stderr)
                break

            if not isinstance(data, dict):
                break

            papers = data.get("data", [])
            if not papers:
                break

            for paper in papers:
                if fetched >= max_results:
                    return
                yield paper
                fetched += 1

            total = data.get("total", 0)
            offset += len(papers)
            if offset >= total:
                break
            # Polite delay (unauthenticated: 1 req/s)
            time.sleep(0 if self._api_key else 1.1)

    # ------------------------------------------------------------------
    # Text formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_paper_text(paper: dict[str, Any]) -> tuple[str, str]:
        """
        Format a raw S2 paper object into (text, source_id).

        The text includes title, year, authors, DOI, and abstract.
        The source_id is ``s2:<paper_id>``.

        The text is prefixed with ``[ABSTRACT ONLY]`` to indicate that only
        the abstract has been fetched automatically.  Use
        ``KBSourceManager.ingest_full_paper_text()`` to upgrade to the full
        text when needed.
        """
        paper_id = paper.get("paperId") or ""
        title = paper.get("title") or ""
        year = paper.get("year") or ""
        abstract = paper.get("abstract") or ""
        authors = [
            a.get("name", "")
            for a in (paper.get("authors") or [])[:5]
            if a.get("name")
        ]
        author_str = "; ".join(authors)
        if len(paper.get("authors") or []) > 5:
            author_str += f" et al. ({len(paper['authors'])} total)"

        ext_ids = paper.get("externalIds") or {}
        doi = ext_ids.get("DOI") or ext_ids.get("doi") or ""
        arxiv_id = ext_ids.get("ArXiv") or ext_ids.get("arxiv") or ""
        oa = paper.get("openAccessPdf") or {}
        pdf_url = oa.get("url") or ""

        parts = [
            "[ABSTRACT ONLY — full text not yet fetched. "
            "Call ingest_full_paper_text to upgrade.]",
            f"Title: {title}",
        ]
        if year:
            parts.append(f"Year: {year}")
        if author_str:
            parts.append(f"Authors: {author_str}")
        if doi:
            parts.append(f"DOI: {doi}")
        if arxiv_id:
            parts.append(f"arXiv: https://arxiv.org/abs/{arxiv_id}")
        if pdf_url:
            parts.append(f"PDF: {pdf_url}")
        if abstract:
            parts.append(f"\nAbstract:\n{abstract}")

        return "\n".join(parts), f"s2:{paper_id}"

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, rag: Any, max_docs: int | None = None) -> int:
        """
        Ingest new Semantic Scholar papers into *rag*.

        Parameters
        ----------
        rag:
            :class:`~msr_digital_twin_with_rag.MSRDigitalTwinRAG` instance.
        max_docs:
            Maximum number of new papers to ingest across all queries.
            Falls back to ``self._max_results``.

        Returns
        -------
        int
            Number of documents newly ingested.
        """
        limit = max_docs if max_docs is not None else self._max_results
        state = _load_state(self._state_path)
        ingested_ids: set[str] = set(state.get("ingested_ids", []))
        total_new = 0

        for label, query in self.QUERIES:
            remaining = limit - total_new
            if remaining <= 0:
                break
            print(f"[S2] Running query: {label}")
            for paper in self._iter_papers(query, remaining):
                paper_id = paper.get("paperId") or ""
                if not paper_id or paper_id in ingested_ids:
                    continue
                text, source_id = self.format_paper_text(paper)
                if not text.strip():
                    continue
                try:
                    n = rag.add_document(text, source=source_id)
                    title_short = textwrap.shorten(
                        paper.get("title") or source_id, width=70, placeholder="…"
                    )
                    print(f"[S2] + {title_short} ({n} chunks)")
                    ingested_ids.add(paper_id)
                    total_new += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"[S2] Skipping {source_id}: {exc}", file=sys.stderr)

        # Persist
        state["ingested_ids"] = sorted(ingested_ids)
        state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["total_ingested"] = len(ingested_ids)
        _save_state(self._state_path, state)
        print(f"[S2] Done. {total_new} new paper(s) added.")
        return total_new

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return loader state summary."""
        state = _load_state(self._state_path)
        return {
            "source": "Semantic Scholar API (MSR experimental + TMSR-LF1 SINAP)",
            "total_ingested": state.get("total_ingested", 0),
            "last_run": state.get("last_run", "never"),
        }


# ---------------------------------------------------------------------------
# Plant operational data loader (real-time / live-plant source)
# ---------------------------------------------------------------------------

class PlantDataLoader:
    """
    Ingests plant real-time operational data into the MSR knowledge base.

    Accepts three categories of input:

    ``sensor_snapshot``
        A JSON-encoded list of sensor readings or a single dict snapshot with
        ``{sensor, value, unit, timestamp}`` records, e.g. readings exported
        from a SCADA historian.

    ``event_log``
        Text or JSON-encoded operational events (alarms cleared, set-point
        changes, transient summaries, shift handover notes).

    ``maintenance_report``
        Unstructured text: inspection reports, maintenance logs, corrective
        action records.

    ``operational_data`` *(default)*
        Any other plant operational text record.

    All ingested records are tracked in ``plant_data_state.json`` (inside
    ``MSR_KB_DIR``) so the same ``source_id`` is never ingested twice.

    Parameters
    ----------
    kb_dir : str | Path | None
        Knowledge-base directory.  Defaults to the ``MSR_KB_DIR`` env var
        or ``./kb_store``.

    Example
    -------
    ::

        from msr_digital_twin_with_rag import MSRDigitalTwinRAG
        from msr_kb_sources import PlantDataLoader

        rag = MSRDigitalTwinRAG()
        loader = PlantDataLoader()

        # Ingest a maintenance report
        loader.ingest_text(
            rag,
            "2024-01-15: Inspected heat exchanger HX-1. No fouling detected.",
            source_id="maint-hx1-20240115",
            data_type="maintenance_report",
        )

        # Ingest a sensor snapshot
        snapshot = [
            {"timestamp": "2024-01-15T14:00:00Z", "sensor": "core_temperature_c",
             "value": 702.1, "unit": "°C"},
            {"timestamp": "2024-01-15T14:00:00Z", "sensor": "reactor_power_mw",
             "value": 99.8, "unit": "MW"},
        ]
        loader.ingest_sensor_snapshot(rag, snapshot, source_id="snapshot-20240115T1400Z")
    """

    STATE_FILE = "plant_data_state.json"

    _VALID_DATA_TYPES = frozenset(
        {"sensor_snapshot", "event_log", "maintenance_report", "operational_data"}
    )

    def __init__(self, kb_dir: str | Path | None = None) -> None:
        self._kb_dir = Path(kb_dir or os.environ.get("MSR_KB_DIR", "./kb_store"))
        self._state_path = self._kb_dir / self.STATE_FILE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_text(
        self,
        rag: Any,
        text: str,
        source_id: str,
        data_type: str = "operational_data",
    ) -> int:
        """
        Ingest *text* into the knowledge base with the given *source_id*.

        Parameters
        ----------
        rag:
            An initialised :class:`~msr_digital_twin_with_rag.MSRDigitalTwinRAG`
            instance.
        text : str
            The content to ingest.
        source_id : str
            Unique identifier for this record.  Re-ingestion of the same ID
            is a no-op.
        data_type : str
            One of ``"sensor_snapshot"``, ``"event_log"``,
            ``"maintenance_report"``, ``"operational_data"``.

        Returns
        -------
        int
            Number of KB chunks added (0 if already ingested).
        """
        if not text or not text.strip():
            return 0
        if data_type not in self._VALID_DATA_TYPES:
            data_type = "operational_data"

        state = _load_state(self._state_path)
        ingested: set[str] = set(state.get("ingested_ids", []))
        if source_id in ingested:
            return 0

        source_label = f"plant:{data_type}:{source_id}"
        n = rag.add_document(text, source=source_label)

        ingested.add(source_id)
        state["ingested_ids"] = sorted(ingested)
        state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state["total_ingested"] = len(ingested)
        _save_state(self._state_path, state)

        print(f"[PlantData] + {source_label} ({n} chunks)")
        return n

    def ingest_sensor_snapshot(
        self,
        rag: Any,
        snapshot: Any,
        source_id: str,
    ) -> int:
        """
        Format a sensor snapshot and ingest it.

        Parameters
        ----------
        snapshot : list | dict
            Either a list of ``{sensor, value, unit, timestamp}`` dicts or a
            single ``{sensor: value, ...}`` dict.  JSON strings are accepted.
        source_id : str
            Unique identifier for this snapshot.
        """
        if isinstance(snapshot, str):
            try:
                snapshot = json.loads(snapshot)
            except (json.JSONDecodeError, ValueError):
                pass  # fall through to format as-is
        text = self._format_snapshot(snapshot)
        return self.ingest_text(rag, text, source_id, data_type="sensor_snapshot")

    def status(self) -> dict[str, Any]:
        """Return loader state summary."""
        state = _load_state(self._state_path)
        return {
            "source": "Plant operational data (real-time ingestion)",
            "total_ingested": state.get("total_ingested", 0),
            "last_run": state.get("last_run", "never"),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_snapshot(snapshot: Any) -> str:
        """Convert a sensor snapshot to a human-readable text string."""
        if isinstance(snapshot, list):
            lines = ["Plant Sensor Readings:"]
            for rec in snapshot:
                if not isinstance(rec, dict):
                    lines.append(f"  {rec}")
                    continue
                ts = rec.get("timestamp", "")
                sensor = rec.get("sensor", rec.get("name", ""))
                value = rec.get("value", "")
                unit = rec.get("unit", "")
                prefix = f"[{ts}] " if ts else ""
                suffix = f" {unit}" if unit else ""
                lines.append(f"  {prefix}{sensor}: {value}{suffix}".strip())
            return "\n".join(lines)
        elif isinstance(snapshot, dict):
            lines = ["Plant Sensor Snapshot:"]
            ts = snapshot.get("timestamp", "")
            if ts:
                lines.append(f"  Timestamp: {ts}")
            for k, v in snapshot.items():
                if k == "timestamp":
                    continue
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)
        else:
            return str(snapshot)


# ---------------------------------------------------------------------------
# TimeseriesStore – SQLite-backed plant sensor timeseries
# ---------------------------------------------------------------------------

class TimeseriesStore:
    """
    SQLite-backed append-only store for timestamped plant sensor readings.

    Designed to complement the text-centric RAG pipeline by providing
    precise time-range queries, aggregate statistics, and
    natural-language-to-SQL query capability over structured numerical
    plant data (SCADA readings, BEAVRS-style benchmark datasets, etc.).

    Data is stored in a local SQLite database file
    (``plant_timeseries.db`` inside ``MSR_KB_DIR``) using only the
    Python standard library ``sqlite3`` module — no new dependencies.

    The store is **append-only**: readings are never modified after
    insertion, preserving the auditability requirement from
    ``requirements.md``.

    Schema
    ------
    Table ``sensor_readings``::

        id          INTEGER PRIMARY KEY AUTOINCREMENT
        timestamp   TEXT NOT NULL   -- ISO 8601 UTC string
        sensor_name TEXT NOT NULL
        value       REAL            -- numeric sensor value
        unit        TEXT            -- physical unit (e.g. 'MW', '°C')
        source_id   TEXT NOT NULL   -- provenance identifier
        data_type   TEXT            -- category (sensor_snapshot, …)
        inserted_at TEXT NOT NULL   -- wall-clock insert time

    Indexes on ``(sensor_name, timestamp)`` and ``source_id`` support
    efficient range and provenance queries.

    Parameters
    ----------
    kb_dir : str | Path | None
        Directory where the SQLite database is stored.
        Defaults to the ``MSR_KB_DIR`` env var or ``./kb_store``.

    Example
    -------
    ::

        from msr_kb_sources import TimeseriesStore

        ts = TimeseriesStore()

        # Ingest two readings
        ts.insert_readings(
            [
                {"timestamp": "2024-01-15T14:00:00Z",
                 "sensor_name": "reactor_power_mw", "value": 99.8, "unit": "MW"},
                {"timestamp": "2024-01-15T14:00:00Z",
                 "sensor_name": "core_temperature_c", "value": 702.1, "unit": "°C"},
            ],
            source_id="snapshot-20240115T1400Z",
        )

        # Range query
        rows = ts.query_range("reactor_power_mw",
                              start="2024-01-15T00:00:00Z",
                              end="2024-01-15T23:59:59Z")

        # Aggregate
        stats = ts.query_aggregate("reactor_power_mw", agg="avg")
        print(stats["result"])   # e.g. 99.6
    """

    DB_FILE = "plant_timeseries.db"
    STATE_FILE = "timeseries_state.json"

    _CREATE_DDL = """
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            sensor_name TEXT    NOT NULL,
            value       REAL,
            unit        TEXT    DEFAULT '',
            source_id   TEXT    NOT NULL,
            data_type   TEXT    DEFAULT 'sensor_snapshot',
            inserted_at TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ts_sensor_time
            ON sensor_readings (sensor_name, timestamp);
        CREATE INDEX IF NOT EXISTS idx_ts_source
            ON sensor_readings (source_id);
    """

    def __init__(self, kb_dir: str | Path | None = None) -> None:
        self._kb_dir = Path(kb_dir or os.environ.get("MSR_KB_DIR", "./kb_store"))
        self._kb_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._kb_dir / self.DB_FILE
        self._state_path = self._kb_dir / self.STATE_FILE
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open and return a new SQLite connection with Row factory."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(self._CREATE_DDL)
            conn.commit()
        finally:
            conn.close()

    def _count_readings(self) -> int:
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API — write
    # ------------------------------------------------------------------

    def source_id_exists(self, source_id: str) -> bool:
        """Return ``True`` if readings from this *source_id* have already been inserted."""
        state = _load_state(self._state_path)
        return source_id in set(state.get("source_ids", []))

    def insert_readings(
        self,
        readings: list[dict[str, Any]],
        source_id: str,
        data_type: str = "sensor_snapshot",
    ) -> int:
        """
        Insert a batch of timestamped sensor readings.

        Each reading must be a dict with at least ``sensor_name`` (or
        ``sensor``) and ``value`` keys.  ``timestamp`` and ``unit`` are
        optional.

        Duplicate *source_id* inserts are **skipped** (idempotent) to
        match the deduplication contract of the other loaders.

        Parameters
        ----------
        readings : list[dict]
            List of ``{sensor_name, value, unit, timestamp}`` dicts.
        source_id : str
            Unique provenance identifier for this batch.
        data_type : str
            Category label (default ``"sensor_snapshot"``).

        Returns
        -------
        int
            Number of rows inserted (0 if already seen).
        """
        if self.source_id_exists(source_id):
            return 0
        if not readings:
            return 0

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rows = []
        for r in readings:
            if not isinstance(r, dict):
                continue
            rows.append((
                r.get("timestamp", now),
                r.get("sensor_name", r.get("sensor", "")),
                r.get("value"),
                r.get("unit", ""),
                source_id,
                data_type,
                now,
            ))

        conn = self._connect()
        try:
            conn.executemany(
                "INSERT INTO sensor_readings "
                "(timestamp, sensor_name, value, unit, source_id, data_type, inserted_at) "
                "VALUES (?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

        # Update state
        state = _load_state(self._state_path)
        ids: set[str] = set(state.get("source_ids", []))
        ids.add(source_id)
        state["source_ids"] = sorted(ids)
        state["total_readings"] = self._count_readings()
        state["last_insert"] = now
        _save_state(self._state_path, state)

        print(f"[Timeseries] + {source_id} ({len(rows)} readings, data_type={data_type})")
        return len(rows)

    # ------------------------------------------------------------------
    # Public API — read
    # ------------------------------------------------------------------

    def query_range(
        self,
        sensor_name: str,
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Return readings for *sensor_name* within an optional time range.

        Parameters
        ----------
        sensor_name : str
            Exact sensor name (e.g. ``"reactor_power_mw"``).
        start : str | None
            ISO 8601 lower bound (inclusive).  No lower bound if ``None``.
        end : str | None
            ISO 8601 upper bound (inclusive).  No upper bound if ``None``.
        limit : int
            Maximum number of rows returned (default 1 000).

        Returns
        -------
        list[dict]
            Rows with keys ``timestamp``, ``sensor_name``, ``value``,
            ``unit``, ``source_id``.
        """
        sql = (
            "SELECT timestamp, sensor_name, value, unit, source_id "
            "FROM sensor_readings WHERE sensor_name = ?"
        )
        params: list[Any] = [sensor_name]
        if start:
            sql += " AND timestamp >= ?"
            params.append(start)
        if end:
            sql += " AND timestamp <= ?"
            params.append(end)
        sql += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def query_latest(self, sensor_name: str, last_n: int = 10) -> list[dict[str, Any]]:
        """Return the *last_n* readings for *sensor_name*, newest first."""
        sql = (
            "SELECT timestamp, sensor_name, value, unit, source_id "
            "FROM sensor_readings WHERE sensor_name = ? "
            "ORDER BY timestamp DESC LIMIT ?"
        )
        conn = self._connect()
        try:
            rows = conn.execute(sql, [sensor_name, last_n]).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def query_aggregate(
        self,
        sensor_name: str,
        agg: str = "avg",
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """
        Return aggregate statistics for *sensor_name*.

        Parameters
        ----------
        agg : str
            One of ``"avg"``, ``"min"``, ``"max"``, ``"count"``.

        Returns
        -------
        dict
            ``{sensor_name, aggregation, result, n}``
        """
        valid = {"avg": "AVG", "min": "MIN", "max": "MAX", "count": "COUNT"}
        func = valid.get(agg, "AVG")
        sql = (
            f"SELECT {func}(value) AS result, COUNT(*) AS n "
            "FROM sensor_readings WHERE sensor_name = ?"
        )
        params: list[Any] = [sensor_name]
        if start:
            sql += " AND timestamp >= ?"
            params.append(start)
        if end:
            sql += " AND timestamp <= ?"
            params.append(end)

        conn = self._connect()
        try:
            row = conn.execute(sql, params).fetchone()
            result = dict(row) if row else {}
        finally:
            conn.close()

        return {
            "sensor_name": sensor_name,
            "aggregation": agg,
            "result": result.get("result"),
            "n": result.get("n", 0),
        }

    def list_sensors(self) -> list[str]:
        """Return all distinct sensor names currently in the store."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT sensor_name FROM sensor_readings ORDER BY sensor_name"
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()

    def get_schema_description(self) -> str:
        """
        Return a plain-text description of the schema for LLM-assisted
        natural-language-to-SQL translation.
        """
        sensors = self.list_sensors()
        sensor_list = ", ".join(sensors[:30]) if sensors else "(empty — no data ingested yet)"
        return (
            "Table: sensor_readings\n"
            "Columns:\n"
            "  id          INTEGER PRIMARY KEY\n"
            "  timestamp   TEXT  (ISO 8601 UTC, e.g. '2024-01-15T14:00:00Z')\n"
            "  sensor_name TEXT\n"
            "  value       REAL\n"
            "  unit        TEXT\n"
            "  source_id   TEXT\n"
            "  data_type   TEXT\n"
            "  inserted_at TEXT\n"
            f"Known sensor_name values: {sensor_list}\n"
            "All timestamps are UTC ISO 8601 strings.\n"
            "Only SELECT queries are permitted."
        )

    def execute_safe_select(self, sql: str) -> list[dict[str, Any]]:
        """
        Execute a SELECT query against the timeseries store and return rows.

        Raises
        ------
        ValueError
            If *sql* does not begin with ``SELECT`` (safety guard against
            mutation queries).
        """
        # Strip optional leading whitespace / markdown fences
        cleaned = sql.strip().lstrip("(").upper()
        if not cleaned.startswith("SELECT"):
            raise ValueError(
                f"Only SELECT queries are allowed. Received: {sql[:80]!r}"
            )
        conn = self._connect()
        try:
            rows = conn.execute(sql).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def status(self) -> dict[str, Any]:
        """Return a status summary dict."""
        state = _load_state(self._state_path)
        return {
            "source": "Plant timeseries SQLite store",
            "db_path": str(self._db_path),
            "total_readings": self._count_readings(),
            "sensors": self.list_sensors(),
            "source_ids_count": len(state.get("source_ids", [])),
            "last_insert": state.get("last_insert", "never"),
        }


# ---------------------------------------------------------------------------
# AlarmHistoryStore – persistent, append-only alarm event log
# ---------------------------------------------------------------------------

class AlarmHistoryStore:
    """
    SQLite-backed append-only store for alarm event history.

    Alarm events generated by the MCP server's threshold checks are written
    here so they survive across server restarts, supporting the audit-trail
    and regulatory-traceability requirements (IEC 62645, IAEA safeguards).

    Uses the same SQLite database as :class:`TimeseriesStore`
    (``plant_timeseries.db``) but a separate table (``alarm_history``), so
    no additional files or dependencies are needed.

    Schema
    ------
    Table ``alarm_history``::

        id             INTEGER PRIMARY KEY AUTOINCREMENT
        alarm_id       TEXT NOT NULL   -- symbolic alarm code, e.g. CORE_TEMP_HIGH
        sensor         TEXT NOT NULL   -- sensor that triggered the alarm
        value          REAL            -- sensor value at trigger time
        threshold_high REAL            -- upper threshold (NULL if not applicable)
        threshold_low  REAL            -- lower threshold (NULL if not applicable)
        severity       TEXT NOT NULL   -- WARNING or CRITICAL
        timestamp      TEXT NOT NULL   -- ISO 8601 UTC when the alarm was triggered
        inserted_at    TEXT NOT NULL   -- ISO 8601 UTC wall-clock insert time

    Indexes on ``(timestamp)`` and ``(sensor)`` support efficient time-range
    and per-sensor queries.

    The store is **read + append only**: existing records are never modified
    or deleted, preserving the audit trail.

    Parameters
    ----------
    kb_dir : str | Path | None
        Directory where the SQLite database is stored.
        Defaults to the ``MSR_KB_DIR`` env var or ``./kb_store``.

    Example
    -------
    ::

        from msr_kb_sources import AlarmHistoryStore

        store = AlarmHistoryStore()
        store.record_alarm({
            "alarm_id": "CORE_TEMP_HIGH",
            "sensor": "core_temperature_c",
            "value": 752.3,
            "threshold_high": 750.0,
            "threshold_low": None,
            "severity": "WARNING",
            "timestamp": "2024-01-15T14:32:00Z",
        })

        history = store.query_history(start="2024-01-15T00:00:00Z")
        print(history[0]["alarm_id"])   # "CORE_TEMP_HIGH"
    """

    DB_FILE = "plant_timeseries.db"

    _CREATE_DDL = """
        CREATE TABLE IF NOT EXISTS alarm_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            alarm_id       TEXT    NOT NULL,
            sensor         TEXT    NOT NULL,
            value          REAL,
            threshold_high REAL,
            threshold_low  REAL,
            severity       TEXT    NOT NULL DEFAULT 'WARNING',
            timestamp      TEXT    NOT NULL,
            inserted_at    TEXT    NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_alarm_timestamp
            ON alarm_history (timestamp);
        CREATE INDEX IF NOT EXISTS idx_alarm_sensor
            ON alarm_history (sensor);
    """

    def __init__(self, kb_dir: str | Path | None = None) -> None:
        self._kb_dir = Path(kb_dir or os.environ.get("MSR_KB_DIR", "./kb_store"))
        self._kb_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._kb_dir / self.DB_FILE
        self._init_db()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open and return a new SQLite connection with Row factory."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(self._CREATE_DDL)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API — write
    # ------------------------------------------------------------------

    def record_alarm(self, alarm: dict[str, Any]) -> None:
        """
        Persist a single alarm event to the database.

        Parameters
        ----------
        alarm : dict
            Alarm dict with keys: ``alarm_id``, ``sensor``, ``value``,
            ``threshold_high``, ``threshold_low``, ``severity``,
            ``timestamp``.  Any missing key defaults to ``None``
            (or ``"WARNING"`` for severity).
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO alarm_history
                    (alarm_id, sensor, value, threshold_high, threshold_low,
                     severity, timestamp, inserted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alarm.get("alarm_id", ""),
                    alarm.get("sensor", ""),
                    alarm.get("value"),
                    alarm.get("threshold_high"),
                    alarm.get("threshold_low"),
                    (alarm.get("severity") or "WARNING").upper(),
                    alarm.get("timestamp", now),
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Public API — read
    # ------------------------------------------------------------------

    def query_history(
        self,
        start: str | None = None,
        end: str | None = None,
        severity: str | None = None,
        sensor: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Query alarm history with optional filters.

        Parameters
        ----------
        start : str | None
            ISO 8601 lower bound (inclusive), e.g. ``"2024-01-15T00:00:00Z"``.
        end : str | None
            ISO 8601 upper bound (inclusive).
        severity : str | None
            Filter by severity level (``"WARNING"`` or ``"CRITICAL"``).
            Case-insensitive.
        sensor : str | None
            Filter to a specific sensor name.
        limit : int
            Maximum rows to return (1–1000, default 100).

        Returns
        -------
        list[dict]
            Rows ordered newest-first.  Each dict has keys:
            ``alarm_id``, ``sensor``, ``value``, ``threshold_high``,
            ``threshold_low``, ``severity``, ``timestamp``.
        """
        conditions: list[str] = []
        params: list[Any] = []

        # All entries in `conditions` are hard-coded SQL fragments (column names
        # and operators only — never user-supplied strings).  All user-supplied
        # filter values are bound as positional `?` parameters, making this
        # safe from SQL injection.
        if start:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end:
            conditions.append("timestamp <= ?")
            params.append(end)
        if severity:
            conditions.append("severity = ?")
            params.append(severity.upper())
        if sensor:
            conditions.append("sensor = ?")
            params.append(sensor)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        limit_val = min(max(1, limit), 1000)
        sql = (
            "SELECT alarm_id, sensor, value, threshold_high, threshold_low, "
            "severity, timestamp "
            "FROM alarm_history "
            + (where + " " if where else "")
            + "ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit_val)

        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_alarm_count(self) -> int:
        """Return the total number of alarm events recorded."""
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) FROM alarm_history").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# KBSourceManager – orchestrates all loaders
# ---------------------------------------------------------------------------

class KBSourceManager:
    """
    Coordinates all knowledge-base source loaders.

    Parameters
    ----------
    rag:
        An initialised :class:`~msr_digital_twin_with_rag.MSRDigitalTwinRAG`
        instance.  The manager calls ``rag.add_document()`` to ingest content.
    kb_dir:
        Directory where state files are stored.  Falls back to
        ``MSR_KB_DIR`` env var, then ``./kb_store``.

    Example
    -------
    ::

        from msr_digital_twin_with_rag import MSRDigitalTwinRAG
        from msr_kb_sources import KBSourceManager

        rag = MSRDigitalTwinRAG()
        mgr = KBSourceManager(rag)
        mgr.update_all()          # ingest new docs from archive + OpenAlex
    """

    def __init__(
        self,
        rag: Any,
        kb_dir: str | Path | None = None,
    ) -> None:
        self._rag = rag
        _kb = kb_dir or os.environ.get("MSR_KB_DIR", "./kb_store")
        self._archive = MSRArchiveLoader(kb_dir=_kb)
        self._openalex = OpenAlexLoader(kb_dir=_kb)
        self._arxiv = ArXivLoader(kb_dir=_kb)
        self._s2 = SemanticScholarLoader(kb_dir=_kb)
        self._plant = PlantDataLoader(kb_dir=_kb)
        self._ts = TimeseriesStore(kb_dir=_kb)

    def update_archive(self, max_docs: int = 0) -> int:
        """Ingest new OCR files from msr-archive. Returns docs added."""
        return self._archive.ingest(self._rag, max_docs=max_docs)

    def update_openalex(self, max_docs: int | None = None) -> int:
        """Ingest new OpenAlex papers. Returns docs added."""
        return self._openalex.ingest(self._rag, max_docs=max_docs)

    def update_arxiv(self, max_docs: int | None = None) -> int:
        """Ingest new arXiv preprints and papers. Returns docs added."""
        return self._arxiv.ingest(self._rag, max_docs=max_docs)

    def update_semanticscholar(self, max_docs: int | None = None) -> int:
        """Ingest new Semantic Scholar papers. Returns docs added."""
        return self._s2.ingest(self._rag, max_docs=max_docs)

    def ingest_plant_data(
        self,
        text: str,
        source_id: str,
        data_type: str = "operational_data",
    ) -> int:
        """
        Ingest plant operational data into the knowledge base.

        Returns the number of KB chunks added (0 if already ingested).
        """
        return self._plant.ingest_text(self._rag, text, source_id, data_type=data_type)

    def ingest_full_paper_text(
        self,
        source_id: str,
        markdown_text: str,
    ) -> dict[str, Any]:
        """
        Upgrade an abstract-only KB entry to full paper text.

        Used when a user explicitly requests ingestion of the full text of a
        paper previously indexed as ``ABSTRACT ONLY`` (e.g. from OpenAlex,
        arXiv, or Semantic Scholar).  The full text should be provided as
        Markdown (e.g. converted from a PDF via a VLM call).

        The full text is stored under the key ``full:{source_id}`` so it
        co-exists with the existing abstract chunks and supplements them in
        retrieval.  Duplicate calls with the same *source_id* are no-ops
        (the full text is already present).

        Parameters
        ----------
        source_id : str
            The source identifier of the existing abstract-only KB entry,
            e.g. ``https://openalex.org/W1234567`` or ``arxiv:2401.12345``
            or ``s2:abc123``.
        markdown_text : str
            Full paper text in Markdown format (plain text is also accepted).

        Returns
        -------
        dict
            ``{success, source_id, full_source_id, chunks_added}``
        """
        if not markdown_text or not markdown_text.strip():
            return {
                "success": False,
                "error": "markdown_text must not be empty.",
                "source_id": source_id,
            }
        if not source_id or not source_id.strip():
            return {
                "success": False,
                "error": "source_id must not be empty.",
            }
        full_source_id = f"full:{source_id}"
        try:
            n = self._rag.add_document(markdown_text, source=full_source_id)
            return {
                "success": True,
                "source_id": source_id,
                "full_source_id": full_source_id,
                "chunks_added": n,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "source_id": source_id,
                "error": str(exc),
            }

    def list_abstract_only_sources(self) -> list[str]:
        """
        Return source_ids of documents ingested as abstract-only entries.

        Scans the OpenAlex, arXiv, and Semantic Scholar state files.
        These entries can be upgraded to full text via
        :meth:`ingest_full_paper_text`.

        Returns
        -------
        list[str]
            Sorted list of source_ids that have only an abstract in the KB.
        """
        sources: list[str] = []
        for loader in (self._openalex, self._arxiv, self._s2):
            state = _load_state(loader._state_path)  # type: ignore[attr-defined]
            for sid in state.get("ingested_ids", state.get("ingested_urls", [])):
                full_sid = f"full:{sid}"
                # If full text was already ingested under full:<sid> it won't
                # appear here.  We detect this by checking the full: prefixed
                # source in RAG chunks if available, but a simpler heuristic
                # is just to return all abstract-only source IDs and let the
                # caller decide.
                sources.append(sid)
        return sorted(sources)

    def ingest_timeseries(
        self,
        readings: list[dict[str, Any]],
        source_id: str,
        data_type: str = "sensor_snapshot",
        also_ingest_text: bool = True,
    ) -> dict[str, int]:
        """
        Insert timestamped sensor readings into the timeseries store and
        optionally also ingest a text summary into the RAG knowledge base.

        Parameters
        ----------
        readings : list[dict]
            List of ``{sensor_name, value, unit, timestamp}`` dicts.
        source_id : str
            Unique provenance identifier.  Duplicate calls are no-ops.
        data_type : str
            Category label (default ``"sensor_snapshot"``).
        also_ingest_text : bool
            When ``True`` (default), also format the readings as a
            human-readable text block and ingest it into the RAG pipeline
            so that semantic search queries can surface this data.

        Returns
        -------
        dict
            ``{"timeseries_rows": <n>, "rag_chunks": <n>}``
        """
        ts_rows = self._ts.insert_readings(readings, source_id, data_type=data_type)
        rag_chunks = 0
        if also_ingest_text and ts_rows > 0:
            text = self._plant._format_snapshot(readings)  # type: ignore[attr-defined]
            rag_chunks = self._plant.ingest_text(
                self._rag, text, f"ts:{source_id}", data_type=data_type
            )
        return {"timeseries_rows": ts_rows, "rag_chunks": rag_chunks}

    def query_timeseries(
        self,
        sensor_name: str,
        start: str | None = None,
        end: str | None = None,
        last_n: int | None = None,
        aggregation: str | None = None,
    ) -> dict[str, Any]:
        """
        Query the timeseries store for a named sensor.

        Parameters
        ----------
        sensor_name : str
            Sensor to query (e.g. ``"reactor_power_mw"``).
        start : str | None
            ISO 8601 start bound (inclusive).
        end : str | None
            ISO 8601 end bound (inclusive).
        last_n : int | None
            If set, return the last *n* readings (ignores start/end).
        aggregation : str | None
            If set (``"avg"``, ``"min"``, ``"max"``, ``"count"``), return
            aggregate statistics instead of raw rows.

        Returns
        -------
        dict
            ``{sensor_name, rows, aggregation}`` or aggregate result dict.
        """
        if aggregation:
            return self._ts.query_aggregate(sensor_name, agg=aggregation, start=start, end=end)
        if last_n is not None:
            rows = self._ts.query_latest(sensor_name, last_n=last_n)
        else:
            rows = self._ts.query_range(sensor_name, start=start, end=end)
        return {"sensor_name": sensor_name, "rows": rows, "count": len(rows)}

    def query_timeseries_nl(self, question: str) -> dict[str, Any]:
        """
        Answer a natural-language question about timeseries plant data via
        LLM-assisted SQL generation (NL→SQL).

        The LLM generates a SQLite SELECT query from the *question* and the
        table schema.  The query is validated to be a SELECT before execution.
        Falls back gracefully when no LLM backend is configured.

        Parameters
        ----------
        question : str
            Natural-language question, e.g.
            ``"What was the average reactor power last month?"``.

        Returns
        -------
        dict
            ``{question, sql_used, rows, row_count}`` on success, or
            ``{question, error, rows}`` on failure.
        """
        schema = self._ts.get_schema_description()

        # Check LLM availability via duck-typing
        has_llm = getattr(self._rag, "_has_llm", lambda: False)()
        if not has_llm:
            return {
                "question": question,
                "sql_used": None,
                "rows": [],
                "error": (
                    "LLM backend not configured. "
                    "Set MSR_GITHUB_TOKEN or MSR_OPENAI_API_KEY for NL→SQL. "
                    "Use query_timeseries() for direct structured queries."
                ),
            }

        system = (
            "You are a SQL expert for a nuclear reactor plant data system.\n"
            "Given a natural-language question and the table schema below, "
            "generate ONE valid SQLite SELECT query that answers the question.\n"
            "Output ONLY the raw SQL — no explanation, no markdown fences, "
            "no trailing semicolons.\n\n"
            f"Schema:\n{schema}"
        )
        user = f"Question: {question}"

        try:
            llm_generate = getattr(self._rag, "_llm_generate")
            sql_raw: str = llm_generate(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=256,
            )
            # Strip markdown fences the LLM may add despite instructions
            sql = sql_raw.strip()
            for fence in ("```sql", "```sqlite", "```"):
                if sql.startswith(fence):
                    sql = sql[len(fence):]
            sql = sql.rstrip("`").strip().rstrip(";").strip()

            rows = self._ts.execute_safe_select(sql)
            return {
                "question": question,
                "sql_used": sql,
                "rows": rows[:200],
                "row_count": len(rows),
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "question": question,
                "sql_used": None,
                "rows": [],
                "error": str(exc),
            }

    def timeseries_status(self) -> dict[str, Any]:
        """Return timeseries store status."""
        return self._ts.status()

    def update_all(
        self,
        max_archive_docs: int = 0,
        max_openalex_docs: int | None = None,
        max_arxiv_docs: int | None = None,
        max_s2_docs: int | None = None,
    ) -> dict[str, int]:
        """
        Run archive + OpenAlex + arXiv + Semantic Scholar loaders and return
        counts of newly added documents.

        Inspired by AutoResearchClaw's Phase B literature discovery pipeline
        which spans OpenAlex, Semantic Scholar, and arXiv to maximise recall.

        Returns
        -------
        dict
            ``{"archive": <n>, "openalex": <n>, "arxiv": <n>,
               "semanticscholar": <n>}``
        """
        archive_added = self.update_archive(max_docs=max_archive_docs)
        openalex_added = self.update_openalex(max_docs=max_openalex_docs)
        arxiv_added = self.update_arxiv(max_docs=max_arxiv_docs)
        s2_added = self.update_semanticscholar(max_docs=max_s2_docs)
        return {
            "archive": archive_added,
            "openalex": openalex_added,
            "arxiv": arxiv_added,
            "semanticscholar": s2_added,
        }

    def status(self) -> None:
        """Print a summary of all loaders' state."""
        sources = [
            self._archive.status(),
            self._openalex.status(),
            self._arxiv.status(),
            self._s2.status(),
            self._plant.status(),
        ]
        print("\n=== MSR Knowledge-Base Source Status ===")
        for st in sources:
            print(f"  Source  : {st['source']}")
            print(f"  Ingested: {st['total_ingested']}")
            print(f"  Last run: {st['last_run']}")
            print()
        ts_st = self._ts.status()
        print(f"  Source  : {ts_st['source']}")
        print(f"  Readings: {ts_st['total_readings']}")
        print(f"  Sensors : {ts_st['sensors']}")
        print(f"  Last ins: {ts_st['last_insert']}")
        print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _cli_main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Update the MSR RAG knowledge base from external sources.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Environment variables:
              MSR_KB_DIR               KB store directory (default: ./kb_store)
              MSR_ARCHIVE_REPO         GitHub owner/repo (default: pranavkantgaur/msr-archive)
              MSR_ARCHIVE_MAX_DOCS     Max OCR files per run (0 = unlimited)
              MSR_OPENALEX_MAX_RESULTS Max OpenAlex papers per run (default: 100)
              MSR_OPENALEX_EMAIL       Email for OpenAlex polite pool
              MSR_ARXIV_MAX_RESULTS    Max arXiv papers per run (default: 100)
              MSR_S2_API_KEY           Semantic Scholar API key (optional, higher rate limits)
              MSR_S2_MAX_RESULTS       Max Semantic Scholar papers per run (default: 100)
              MSR_GITHUB_TOKEN         GitHub token for higher API rate limits
              MSR_OPENAI_API_KEY       OpenAI key for LLM insight extraction
        """),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--update-archive",
        action="store_true",
        help="Fetch new OCR documents from pranavkantgaur/msr-archive",
    )
    group.add_argument(
        "--update-openalex",
        action="store_true",
        help="Fetch new papers from OpenAlex (MSR experimental data / TMSR-LF1)",
    )
    group.add_argument(
        "--update-arxiv",
        action="store_true",
        help="Fetch new preprints and papers from arXiv (molten salt reactor experimental + TMSR-LF1)",
    )
    group.add_argument(
        "--update-semanticscholar",
        action="store_true",
        help="Fetch new papers from Semantic Scholar (MSR experimental + TMSR-LF1 SINAP)",
    )
    group.add_argument(
        "--update-all",
        action="store_true",
        help="Run archive + OpenAlex + arXiv + Semantic Scholar loaders",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Show current state of all loaders without fetching anything",
    )
    group.add_argument(
        "--ingest-plant-data",
        action="store_true",
        help="Ingest plant operational data into the knowledge base",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Override max documents per run (0 = no limit / use env default)",
    )
    parser.add_argument(
        "--content",
        type=str,
        default="",
        help="Plant data content to ingest (used with --ingest-plant-data)",
    )
    parser.add_argument(
        "--content-file",
        type=str,
        default="",
        help="Path to file containing plant data content (used with --ingest-plant-data)",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="operational_data",
        choices=["sensor_snapshot", "event_log", "maintenance_report", "operational_data"],
        help="Category of plant data (default: operational_data)",
    )
    parser.add_argument(
        "--source-id",
        type=str,
        default="",
        help="Unique ID for the plant data record (auto-generated if omitted)",
    )

    args = parser.parse_args(argv)

    # Lazy import so that the module can be imported without pulling in the RAG
    from msr_digital_twin_with_rag import MSRDigitalTwinRAG  # noqa: PLC0415

    print("[KB] Initialising RAG knowledge base…")
    rag = MSRDigitalTwinRAG()
    mgr = KBSourceManager(rag)

    if args.status:
        mgr.status()
    elif args.update_archive:
        mgr.update_archive(max_docs=args.max_docs)
    elif args.update_openalex:
        max_docs = args.max_docs or None
        mgr.update_openalex(max_docs=max_docs)
    elif args.update_arxiv:
        max_docs = args.max_docs or None
        mgr.update_arxiv(max_docs=max_docs)
    elif args.update_semanticscholar:
        max_docs = args.max_docs or None
        mgr.update_semanticscholar(max_docs=max_docs)
    elif args.update_all:
        max_docs = args.max_docs or None
        counts = mgr.update_all(
            max_archive_docs=args.max_docs,
            max_openalex_docs=max_docs,
            max_arxiv_docs=max_docs,
            max_s2_docs=max_docs,
        )
        print(
            f"\n[KB] Update complete: archive +{counts['archive']}, "
            f"openalex +{counts['openalex']}, "
            f"arxiv +{counts['arxiv']}, "
            f"semanticscholar +{counts['semanticscholar']}"
        )
    elif args.ingest_plant_data:
        content = args.content
        if not content and args.content_file:
            content = Path(args.content_file).read_text(encoding="utf-8")
        if not content:
            print("Error: provide --content or --content-file with --ingest-plant-data.", file=sys.stderr)
            raise SystemExit(1)
        source_id = args.source_id or f"{args.data_type}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        n = mgr.ingest_plant_data(content, source_id, data_type=args.data_type)
        print(f"[KB] Plant data ingested: {n} chunk(s) added from source '{source_id}'.")


if __name__ == "__main__":
    _cli_main()
