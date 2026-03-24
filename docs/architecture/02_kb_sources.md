# Knowledge-Base Source Loaders — Architecture

This diagram details the five document-source loaders in `msr_kb_sources.py`
that populate the MSR RAG knowledge base.

---

## Source-loader overview

```mermaid
flowchart TD
    subgraph KBS["msr_kb_sources.py — KBSourceManager + Loaders"]
        direction TB

        KBM["KBSourceManager\nupdate_all() / update_archive()\nupdate_openalex() / update_arxiv()\nupdate_semanticscholar()"]

        subgraph ARCH["MSRArchiveLoader — ORNL static archive"]
            A1["GitHub Contents API\nGET /repos/{owner}/{repo}/contents/ocr\nOptional MSR_GITHUB_TOKEN\n(60 → 5000 req/hr)"]
            A2["Iterate OCR files\n(text/plain, sorted)"]
            A3["Fetch raw content\nraw.githubusercontent.com"]
            A4["Dedup: skip if\npath already in\narchive_state.json"]
            A5["rag.add_document(text,\n  source_id='archive/{path}',\n  data_type='ornl_report')"]
            A1 --> A2 --> A4 --> A3 --> A5
        end

        subgraph OAL["OpenAlexLoader — academic papers"]
            O1["Query 1 (primary)\n'molten salt reactors\nexperimental data'"]
            O2["Query 2 (TMSR)\n'TMSR-LF1 SINAP\nexperimental'"]
            O3["REST pagination\nGET /works?search=...\n&per_page=25\nup to MSR_OPENALEX_MAX_RESULTS"]
            O4["Dedup: skip if\nOpenAlex ID already in\nopenalex_state.json"]
            O5["Format: title + abstract\n+ authors + year + DOI"]
            O6["rag.add_document(text,\n  source_id='openalex/{id}',\n  data_type='academic_paper')"]
            O1 & O2 --> O3 --> O4 --> O5 --> O6
        end

        subgraph AXL["ArXivLoader — preprints"]
            AX1["Query 1\n'molten salt reactor experimental'"]
            AX2["Query 2\n'TMSR-LF1'"]
            AX3["Atom XML feed\nexport.arxiv.org/api/query\nstart=0, max_results=50\n≥ 3 s between requests (ToS)"]
            AX4["Parse XML\nfeedparser-style ET\n(stdlib xml.etree)"]
            AX5["Dedup: skip if\narXiv ID already in\narxiv_state.json"]
            AX6["Format: title + abstract\n+ authors + year + arXiv ID"]
            AX7["rag.add_document(text,\n  source_id='arxiv/{id}',\n  data_type='academic_paper')"]
            AX1 & AX2 --> AX3 --> AX4 --> AX5 --> AX6 --> AX7
        end

        subgraph S2L["SemanticScholarLoader — literature"]
            S1["Query 1\n'molten salt reactor\nexperimental data'"]
            S2["Query 2\n'TMSR-LF1 SINAP\nexperimental'"]
            S3["S2 Graph API\nGET /paper/search\n?query=...&fields=...\n&limit=50\nOptional MSR_S2_API_KEY\n(1 → 100 req/s)"]
            S4["Dedup: skip if\nS2 ID already in\nsemanticscholar_state.json"]
            S5["Format: title + abstract\n+ authors + year + DOI"]
            S6["rag.add_document(text,\n  source_id='s2/{paperId}',\n  data_type='academic_paper')"]
            S1 & S2 --> S3 --> S4 --> S5 --> S6
        end

        subgraph PDL["PlantDataLoader — real-time plant data"]
            P1["ingest_sensor_snapshot()\nAccepts list of sensor readings\n{sensor, value, unit, timestamp}"]
            P2["ingest_text()\nAccepts free-text operational records\n(event logs, maintenance reports,\n characterisation reports)"]
            P3["Format as human-readable\nstructured text block"]
            P4["Dedup: skip if\nsource_id already in\nplant_data_state.json"]
            P5["rag.add_document(text,\n  source_id=source_id,\n  data_type=data_type)"]
            P1 & P2 --> P3 --> P4 --> P5
        end

        KBM --> ARCH & OAL & AXL & S2L
    end

    %% External targets
    EXT_ORNL["pranavkantgaur/msr-archive\nGitHub repository\nOCR/ directory"]
    EXT_OA["api.openalex.org\n(public, rate-limited)"]
    EXT_AX["export.arxiv.org\n(public, ToS: ≥3 s)"]
    EXT_S2["api.semanticscholar.org\n(public/authenticated)"]
    OPS["Operators / agents\n(POST /data/ingest)"]

    RAG["rag.add_document()\nmsr_digital_twin_with_rag.py"]

    EXT_ORNL -->|HTTPS| ARCH
    EXT_OA   -->|REST| OAL
    EXT_AX   -->|XML| AXL
    EXT_S2   -->|REST| S2L
    OPS      -->|push| PDL

    A5 & O6 & AX7 & S6 & P5 --> RAG

    classDef loader fill:#e8f8e8,stroke:#4caf50,color:#000
    classDef ext    fill:#e8f4f8,stroke:#2196f3,color:#000
    classDef rag    fill:#fff8e1,stroke:#ff9800,color:#000
    class ARCH,OAL,AXL,S2L,PDL loader
    class EXT_ORNL,EXT_OA,EXT_AX,EXT_S2,OPS ext
    class RAG rag
```

---

## State-file deduplication

Every loader persists the set of already-ingested document IDs to a JSON
state file so re-running a loader only adds genuinely new content.

```mermaid
sequenceDiagram
    participant Loader
    participant State as state .json
    participant RAG as rag.add_document()
    participant KB as kb_store/

    Loader->>State: load {ingested_ids}
    loop for each fetched document
        Loader->>State: id in ingested_ids?
        alt already seen
            State-->>Loader: skip
        else new document
            Loader->>RAG: add_document(text, source_id)
            RAG->>KB: chunk + embed + persist
            Loader->>State: add id to ingested_ids
        end
    end
    Loader->>State: save updated state
```

---

## CLI reference

```
python msr_kb_sources.py --update-archive
    → MSRArchiveLoader: fetches all new ORNL OCR files

python msr_kb_sources.py --update-openalex
    → OpenAlexLoader: fetches new academic papers

python msr_kb_sources.py --update-arxiv
    → ArXivLoader: fetches new arXiv preprints

python msr_kb_sources.py --update-semanticscholar
    → SemanticScholarLoader: fetches new S2 papers

python msr_kb_sources.py --update-all
    → KBSourceManager.update_all(): runs all four loaders

python msr_kb_sources.py --status
    → prints ingested counts from all state files

python msr_kb_sources.py --ingest-plant-data \
    --content "Core temp 712°C at 14:32" \
    --data-type sensor_snapshot
    → PlantDataLoader: ingests one record from CLI
```

---

## Environment variables

| Variable | Loader | Purpose |
|---|---|---|
| `MSR_GITHUB_TOKEN` | `MSRArchiveLoader` | GitHub API token (60 → 5000 req/hr) |
| `MSR_ARCHIVE_REPO` | `MSRArchiveLoader` | `owner/repo` (default `pranavkantgaur/msr-archive`) |
| `MSR_ARCHIVE_BRANCH` | `MSRArchiveLoader` | Branch to fetch (default `master`) |
| `MSR_ARCHIVE_MAX_DOCS` | `MSRArchiveLoader` | Max files per run (0 = unlimited) |
| `MSR_OPENALEX_MAX_RESULTS` | `OpenAlexLoader` | Max papers per run (default 100) |
| `MSR_OPENALEX_EMAIL` | `OpenAlexLoader` | Polite-pool header for better rate limits |
| `MSR_ARXIV_MAX_RESULTS` | `ArXivLoader` | Max papers per run (default 100) |
| `MSR_S2_API_KEY` | `SemanticScholarLoader` | API key (1 → 100 req/s) |
| `MSR_S2_MAX_RESULTS` | `SemanticScholarLoader` | Max papers per run (default 100) |
| `MSR_KB_DIR` | all loaders | KB store directory (default `./kb_store`) |
