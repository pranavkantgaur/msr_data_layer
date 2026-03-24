# Architecture Diagrams — MSR Data Layer

This directory contains architecture diagrams for all components of the
`msr_data_layer` system.  Diagrams are written in
[Mermaid](https://mermaid.js.org/) and render natively in GitHub, VS Code,
and most modern documentation tools.

---

## Diagram index

| File | Scope | Key content |
|---|---|---|
| [00_end_to_end.md](00_end_to_end.md) | **Full system** | Every component and data flow, end-to-end: external sources → KB loaders → RAG pipeline → MCP server → transport variants → AWS infrastructure → consumers |
| [01_rag_pipeline.md](01_rag_pipeline.md) | RAG pipeline | Ingestion pipeline (chunking → embedding → insight extraction → KB persistence); retrieval pipeline (query decomposition → hybrid search → synthesis) |
| [02_kb_sources.md](02_kb_sources.md) | KB source loaders | MSRArchiveLoader, OpenAlexLoader, ArXivLoader, SemanticScholarLoader, PlantDataLoader; deduplication sequence; CLI reference; env vars |
| [03_mcp_server.md](03_mcp_server.md) | MCP server | Tool surface (class diagram); read-tool and write-tool data flows; stdio transport; HTTP transport; sensor stub values |
| [04_lambda_deployment.md](04_lambda_deployment.md) | AWS Lambda deployment | CloudFormation resource topology (Lambda, API Gateway, S3, EventBridge, IAM, CloudWatch); request and KB-update lifecycles; SAM parameters; build/deploy commands |
| [05_embedding_engines.md](05_embedding_engines.md) | Embedding engines | Engine selection flowchart; OpenAI / GitHub Models / local GPU / random-projection comparison; LLM selection; hybrid retrieval (dense + sparse) |
| [06_physical_ai_integration.md](06_physical_ai_integration.md) | Physical AI integration | Three-stage training-data pipeline (ORNL retrieval → sensor ingest → episode export); 12 robotic areas; data types per area; API usage pattern |

---

## Quick orientation

```
msr_data_layer/
├── msr_kb_sources.py            ← KB source loaders (see 02_kb_sources.md)
├── msr_digital_twin_with_rag.py ← RAG pipeline (see 01_rag_pipeline.md)
├── msr_mcp_server.py            ← MCP tool surface (see 03_mcp_server.md)
├── msr_mcp_server_main.py       ← stdio entry point (see 03_mcp_server.md)
├── msr_digital_twin_client.py   ← Python subprocess client
├── lambda_function.py           ← AWS Lambda handler (see 04_lambda_deployment.md)
├── template.yaml                ← AWS SAM CloudFormation template
├── Dockerfile.gpu               ← GPU container image
├── Makefile                     ← All build/test/deploy targets
└── use_cases/
    └── physical_ai/             ← 12 robotic-area use cases (see 06_physical_ai_integration.md)
```

For a single-page summary of the whole system, start with
[00_end_to_end.md](00_end_to_end.md).

---

## Rendering Mermaid diagrams

**GitHub** — renders automatically in `.md` files in any modern GitHub UI.

**VS Code** — install the
[Mermaid Preview](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid)
extension or use the built-in Markdown preview (VS Code 1.85+).

**CLI** — install the Mermaid CLI:
```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i 00_end_to_end.md -o end_to_end.svg
```
