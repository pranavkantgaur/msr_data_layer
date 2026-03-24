# MCP Server — Tool Surface & Transport Variants

This diagram details the Model Context Protocol (MCP) server implemented
in `msr_mcp_server.py` and the two transport variants that expose it.

---

## MCP tool surface

```mermaid
classDiagram
    class MSRMCPServer {
        +get_reactor_status() dict
        +get_sensor_reading(sensor_name) dict
        +get_all_sensor_readings() dict
        +get_sensor_history(sensor_name, last_n) dict
        +get_active_alarms() dict
        +get_data_source_info() dict
        +ingest_plant_data(content, source_id, data_type) dict
    }

    class PlantDataLayer {
        +_get_current_state() dict
        <<reads MSR_PLANT_DATA_URL or dev stub>>
        +_STUB_STATE: dict
        +_ALARMS: list
        +_SENSOR_HISTORY: dict
    }

    class RAGPipeline {
        <<msr_digital_twin_with_rag.py>>
        +MSRDigitalTwinRAG
        +answer(question) str
        +add_document(text, source_id) None
    }

    class PlantDataLoader {
        <<msr_kb_sources.py>>
        +ingest_text(rag, text, source_id, data_type) None
        +ingest_sensor_snapshot(rag, readings, source_id) None
    }

    MSRMCPServer --> PlantDataLayer : read tools use
    MSRMCPServer --> RAGPipeline : queries via
    MSRMCPServer --> PlantDataLoader : ingest_plant_data calls
```

---

## Read-tool data flow

```mermaid
sequenceDiagram
    participant Agent as LLM Agent / Operator
    participant MCP  as msr_mcp_server.py
    participant Data as Plant data source
    participant RAG  as MSRDigitalTwinRAG

    Agent->>MCP: get_reactor_status()
    MCP->>Data: _get_current_state()
    alt MSR_PLANT_DATA_URL set and reachable
        Data-->>MCP: JSON sensor readings
    else URL unset or unreachable
        Data-->>MCP: _STUB_STATE (dev stub)
    end
    MCP-->>Agent: {status, power, temp, ...}

    Agent->>MCP: get_sensor_reading("core_temperature_c")
    MCP->>Data: _get_current_state()
    Data-->>MCP: full sensor dict
    MCP-->>Agent: {sensor, value, unit, timestamp}

    Agent->>MCP: get_active_alarms()
    MCP-->>Agent: list of active alarm dicts

    Agent->>MCP: get_data_source_info()
    MCP-->>Agent: {url, status, stub_active, ...}
```

---

## Write-tool (ingest) data flow

```mermaid
sequenceDiagram
    participant Op  as Operator / Agent
    participant MCP as msr_mcp_server.py (ingest_plant_data)
    participant PDL as PlantDataLoader
    participant RAG as MSRDigitalTwinRAG
    participant KB  as kb_store/

    Op->>MCP: ingest_plant_data(content, source_id, data_type)
    MCP->>PDL: ingest_text(rag, content, source_id, data_type)
    PDL->>PDL: check plant_data_state.json\n(skip if already seen)
    PDL->>RAG: add_document(text, source_id, data_type)
    RAG->>RAG: chunk + embed + extract insight
    RAG->>KB: persist chunks.json, embeddings.npy, insights.json
    PDL->>PDL: update plant_data_state.json
    MCP-->>Op: {ingested: true, source_id: ...}
```

---

## stdio transport (local / Claude Desktop)

```mermaid
flowchart LR
    subgraph LOCAL["Local deployment  python msr_mcp_server_main.py"]
        direction LR
        MAIN["msr_mcp_server_main.py\nEntry point\nmcp.run(transport='stdio')"]
        SRV["msr_mcp_server.py\nMCP tool registrations\n@mcp.tool() decorators"]
        SDK["mcp SDK\nJSON-RPC 2.0 framing\nstdin / stdout pipes"]
        MAIN --> SDK --> SRV
    end

    CC["Claude Desktop /\nClaude Code"]
    GHC["GitHub Copilot /\nCustom agents\nmsr_digital_twin_client.py"]
    CC  <-->|stdio MCP| SDK
    GHC <-->|stdio MCP\nsubprocess| SDK

    classDef transport fill:#fce4ec,stroke:#e91e63,color:#000
    class SDK transport
```

The `msr_digital_twin_client.py` `MSRDataLayerClient` spawns
`msr_mcp_server_main.py` as a child process and communicates over its
`stdin`/`stdout` — no network port needed.

---

## HTTP transport (Lambda / local-api)

```mermaid
flowchart LR
    subgraph HTTP["HTTP deployment  lambda_function.py / make local-api"]
        direction TB
        APIGW["API Gateway\n(or SAM local)"]
        LF["lambda_function.py\nroutes HTTP → MCP tools"]
        SRV["msr_mcp_server.py\ntool implementations"]
        APIGW --> LF --> SRV
    end

    subgraph ROUTES["HTTP endpoints"]
        R1["POST /mcp\nJSON-RPC 2.0 MCP protocol\n(MCP-capable hosts)"]
        R2["POST /query\nplain-text question\n→ rag.answer()"]
        R3["POST /kb/update\n{source: archive|openalex|\narxiv|semanticscholar|all}"]
        R4["POST /data/ingest\n{content, source_id, data_type}"]
        R5["GET  /health\n→ {status, version, ...}"]
    end

    APIGW <--> R1 & R2 & R3 & R4 & R5

    classDef route fill:#e8f4f8,stroke:#2196f3,color:#000
    class R1,R2,R3,R4,R5 route
```

---

## Sensor stubs (development mode)

When `MSR_PLANT_DATA_URL` is not set, the server uses a built-in stub with
representative TMSR-LF1/MSRE-class FLiBe reactor parameters:

| Sensor | Stub value | Unit |
|---|---|---|
| `reactor_power_mw` | 100.0 | MW |
| `core_temperature_c` | 700.0 | °C |
| `salt_flow_rate_kg_s` | 250.0 | kg/s |
| `fuel_salt_level_pct` | 87.5 | % |
| `coolant_salt_level_pct` | 91.2 | % |
| `neutron_flux_n_cm2_s` | 2.5 × 10¹³ | n/cm²/s |
| `primary_loop_pressure_bar` | 1.1 | bar |
| `heat_exchanger_outlet_c` | 565.0 | °C |
| `turbine_inlet_temp_c` | 540.0 | °C |
| `turbine_output_mwe` | 42.0 | MWe |
| `tritium_production_rate_g_day` | 0.12 | g/day |
| `off_gas_activity_bq_m3` | 3.8 × 10⁶ | Bq/m³ |
