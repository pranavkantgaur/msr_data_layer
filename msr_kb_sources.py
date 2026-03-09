"""
MSR Knowledge-Base Source Loaders

Two document sources feed the MSRDigitalTwinRAG knowledge base:

1. **Static source** – ``pranavkantgaur/msr-archive`` GitHub repository
   OCR text files from the ``ocr/`` directory (transcribed ORNL Molten Salt
   Reactor reports) are fetched over HTTPS and ingested.  The list of files
   is discovered via the GitHub Contents API.

2. **Dynamic source** – OpenAlex academic papers API
   Papers matching "molten salt reactors experimental data" are fetched
   periodically.  A second targeted query focuses on TMSR-LF1 experimental
   data from the TMSR group at SINAP (Shanghai Institute of Applied Physics).

Both loaders maintain a JSON state file in ``MSR_KB_DIR`` (default
``./kb_store``) so that documents already ingested are never processed
twice.  Re-running the updater therefore only adds truly new content.

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

CLI Usage
---------
    python msr_kb_sources.py --update-archive
    python msr_kb_sources.py --update-openalex
    python msr_kb_sources.py --update-all
    python msr_kb_sources.py --status

Python Usage
------------
    from msr_digital_twin_with_rag import MSRDigitalTwinRAG
    from msr_kb_sources import KBSourceManager

    rag = MSRDigitalTwinRAG()
    mgr = KBSourceManager(rag)
    mgr.update_all()
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
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

        parts = [f"Title: {title}"]
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
# KBSourceManager – orchestrates both loaders
# ---------------------------------------------------------------------------

class KBSourceManager:
    """
    Coordinates the two knowledge-base source loaders.

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
        mgr.update_all()          # ingest new docs from both sources
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

    def update_archive(self, max_docs: int = 0) -> int:
        """Ingest new OCR files from msr-archive. Returns docs added."""
        return self._archive.ingest(self._rag, max_docs=max_docs)

    def update_openalex(self, max_docs: int | None = None) -> int:
        """Ingest new OpenAlex papers. Returns docs added."""
        return self._openalex.ingest(self._rag, max_docs=max_docs)

    def update_all(
        self,
        max_archive_docs: int = 0,
        max_openalex_docs: int | None = None,
    ) -> dict[str, int]:
        """
        Run both loaders and return counts of newly added documents.

        Returns
        -------
        dict
            ``{"archive": <n>, "openalex": <n>}``
        """
        archive_added = self.update_archive(max_docs=max_archive_docs)
        openalex_added = self.update_openalex(max_docs=max_openalex_docs)
        return {"archive": archive_added, "openalex": openalex_added}

    def status(self) -> None:
        """Print a summary of both loaders' state."""
        archive_st = self._archive.status()
        openalex_st = self._openalex.status()
        print("\n=== MSR Knowledge-Base Source Status ===")
        for st in (archive_st, openalex_st):
            print(f"  Source : {st['source']}")
            print(f"  Ingested: {st['total_ingested']}")
            print(f"  Last run: {st['last_run']}")
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
              MSR_OPENALEX_MAX_RESULTS Max papers per run (default: 100)
              MSR_OPENALEX_EMAIL       Email for OpenAlex polite pool
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
        "--update-all",
        action="store_true",
        help="Run both loaders (equivalent to --update-archive + --update-openalex)",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Show current state of both loaders without fetching anything",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Override max documents per run (0 = no limit / use env default)",
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
    elif args.update_all:
        counts = mgr.update_all(
            max_archive_docs=args.max_docs,
            max_openalex_docs=args.max_docs or None,
        )
        print(
            f"\n[KB] Update complete: archive +{counts['archive']}, "
            f"openalex +{counts['openalex']}"
        )


if __name__ == "__main__":
    _cli_main()
