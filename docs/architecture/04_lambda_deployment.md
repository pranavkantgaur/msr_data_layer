# AWS Lambda Deployment Architecture

This diagram details the serverless deployment of the MSR Data Layer on AWS,
defined in `template.yaml` and built/deployed via `Makefile`.

---

## Resource topology

```mermaid
flowchart TD
    subgraph AWS["AWS Account — CloudFormation stack: msr-kb-service"]

        subgraph APILAYER["API Layer"]
            APIGW["API Gateway — MSRHttpApi\nHTTP API v2\nCORS: AllowOrigins *\nStage: {Environment}"]
        end

        subgraph COMPUTE["Compute"]
            FUNC["Lambda — MSRKBFunction\npython3.12 / x86_64\nMemory: 1 024 MB\nTimeout: 900 s (15 min)\nX-Ray tracing: Active\nReserved concurrency: none\n(scales to Lambda default)"]
            GFUNC["Lambda — MSRKBGPUFunction\n(optional, Condition: DeployGPUFunction)\nContainer image from ECR\nsentence-transformers + TinyLlama\nECS/EKS/EC2 recommended\nfor GPU workloads"]
        end

        subgraph STORAGE["Storage"]
            S3["S3 — KBStoreBucket\nmsr-kb-store-{AccountId}-{Environment}\nVersioning: Enabled\nEncryption: AES-256\nPublic access: blocked\nOld-version expiry: 30 days\nStores:\n  chunks.json\n  embeddings.npy\n  insights.json\n  tfidf.json\n  *_state.json files"]
        end

        subgraph EVENTS["Event Sources"]
            EB["EventBridge Rule\nSchedule: rate(1 day)\n(configurable KBUpdateSchedule)\n→ invokes FUNC with\n  {source: 'all'}"]
        end

        subgraph SECURITY["Security"]
            IAM["IAM Role — MSRKBFunctionRole\nManaged: AWSLambdaBasicExecutionRole\nInline: s3:GetObject + s3:PutObject\n  on KBStoreBucket/*\nInline: xray:PutTelemetryRecords\n  xray:PutTraceSegments"]
        end

        subgraph OBS["Observability"]
            CW["CloudWatch\nLog group: /aws/lambda/MSRKBFunction\nRetention: 7 days\nX-Ray service map\n+ traces per request"]
        end

        APIGW -->|ALL /{proxy+}| FUNC
        FUNC  <-->|sync_kb_from_s3()\nsync_kb_to_s3()| S3
        EB    -->|scheduled\nEventBridge invoke| FUNC
        FUNC  --> CW
        IAM   -.->|grants| FUNC
        FUNC  -.->|optionally invokes| GFUNC
    end

    %% External callers
    EXT["External consumers\n(agents, operators)\nHTTP/HTTPS"]
    EXT <-->|HTTPS| APIGW

    classDef aws   fill:#fff3e0,stroke:#ff9800,color:#000
    classDef sec   fill:#fce4ec,stroke:#e91e63,color:#000
    classDef store fill:#fafafa,stroke:#607d8b,color:#000
    classDef obs   fill:#e8f4f8,stroke:#2196f3,color:#000
    class FUNC,GFUNC aws
    class IAM sec
    class S3 store
    class CW obs
```

---

## Request lifecycle (warm Lambda)

```mermaid
sequenceDiagram
    participant Client as HTTP Client
    participant GW    as API Gateway
    participant L     as Lambda (warm)
    participant Cache as Module-level _rag_cache
    participant S3    as S3 KBStoreBucket
    participant SCADA as Plant data source

    Client->>GW: POST /query {"question": "..."}
    GW->>L: invoke(event)

    alt First request (cold start)
        L->>S3: sync_kb_from_s3() → /tmp/kb_store/
        L->>Cache: initialise MSRDigitalTwinRAG\n(loads KB from /tmp)
    else Subsequent requests (warm)
        L->>Cache: reuse existing RAG instance
    end

    L->>Cache: rag.answer(question)
    Cache->>SCADA: _get_current_state()\n(or dev stub)
    SCADA-->>Cache: live plant state
    Cache-->>L: final answer text
    L-->>GW: 200 {"answer": "..."}
    GW-->>Client: 200 {"answer": "..."}
```

---

## KB update lifecycle (EventBridge or manual)

```mermaid
sequenceDiagram
    participant EB    as EventBridge (daily)
    participant L     as Lambda
    participant KBS   as KBSourceManager
    participant ORNL  as ORNL Archive
    participant OA    as OpenAlex/arXiv/S2
    participant KB    as /tmp/kb_store/
    participant S3    as S3 KBStoreBucket

    EB->>L: scheduled invoke {source:"all"}
    L->>S3: sync_kb_from_s3()
    S3-->>L: existing KB files → /tmp/kb_store/
    L->>KBS: update_all()
    KBS->>ORNL: fetch new OCR files
    KBS->>OA: fetch new papers
    KBS->>KB: add new chunks + embeddings
    L->>S3: sync_kb_to_s3()
    S3-->>L: 200 OK
    L->>L: log summary {archive:+N, openalex:+M, ...}
```

---

## Deployment variants

```mermaid
flowchart LR
    subgraph DEV["Local Development"]
        LA["SAM local\nmake local-api\nport 3000\nHTTP → lambda_function.py"]
        ST["stdio\npython msr_mcp_server_main.py\nno HTTP port"]
    end

    subgraph CPU["Lambda (CPU)  make deploy"]
        L1["MSRKBFunction\npython3.12\nOpenAI/GitHub Models API\nfor embeddings + LLM"]
    end

    subgraph GPU["GPU Container  make deploy-gpu"]
        L2["MSRKBGPUFunction\nDocker image (ECR)\nsentence-transformers\n+ TinyLlama\nECS/EKS/EC2 recommended"]
    end

    classDef env fill:#e8f8e8,stroke:#4caf50,color:#000
    class DEV,CPU,GPU env
```

---

## SAM parameters

| Parameter | Default | Description |
|---|---|---|
| `Environment` | `prod` | `dev` / `staging` / `prod` — affects resource names |
| `OpenAIApiKey` | *(blank)* | OpenAI key; leave blank to use random-projection |
| `MsrApiKey` | *(blank)* | `X-Api-Key` auth header value; blank = no auth |
| `OpenAlexEmail` | *(blank)* | Polite-pool email for OpenAlex |
| `GithubToken` | *(blank)* | GitHub PAT for msr-archive + GitHub Models |
| `OpenAlexMaxResults` | `100` | Papers fetched per scheduled run |
| `KBUpdateSchedule` | `rate(1 day)` | EventBridge schedule expression |
| `PlantDataUrl` | *(blank)* | External SCADA/historian URL; blank = dev stub |
| `UseLocalGPU` | `false` | Deploy GPU container variant |
| `GPUContainerImageUri` | *(blank)* | ECR URI for GPU image |

---

## Build and deploy commands

```bash
# Build Lambda-compatible package (inside Amazon Linux 2023 Docker container)
make build

# First-time interactive deploy (creates samconfig.toml)
make deploy-guided

# Subsequent deploy (uses saved samconfig.toml)
make deploy

# GPU container build + deploy
make build-gpu-container
make push-gpu-container ECR_REPO=<ecr-uri>
make deploy-gpu ECR_REPO=<ecr-uri>

# Local development server
make local-api       # starts on http://127.0.0.1:3000
make local-health    # GET /health
make local-query     # POST /query (test question)
```
