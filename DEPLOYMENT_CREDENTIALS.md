# MSR Data Layer – Deployment Credentials Checklist

This document lists every credential and configuration value needed to
deploy the MSR data layer to AWS and run the interactive demo notebook.
Provide these values and the deployment can be completed in approximately
15 minutes with `make deploy-guided`.

---

## Required Credentials

### 1. AWS Credentials (for deployment + S3 persistence)

| Credential | How to provide | Notes |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | `aws configure` OR environment variable | IAM user with the permissions below |
| `AWS_SECRET_ACCESS_KEY` | `aws configure` OR environment variable | Same IAM user |
| `AWS_DEFAULT_REGION` | `aws configure` OR environment variable | Recommended: `us-east-1` |

**Required IAM permissions** (attach to the deploying IAM user or role):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": ["cloudformation:*", "s3:*",
        "lambda:*", "apigateway:*", "iam:*", "logs:*",
        "events:*", "xray:*", "ecr:*"],
      "Resource": "*" }
  ]
}
```

A pre-built AWS managed policy combination that covers this:
`AdministratorAccess` (simplest for a sandbox/demo account) or
`AWSCloudFormationFullAccess + AWSLambda_FullAccess + AmazonS3FullAccess +
AmazonAPIGatewayAdministrator + IAMFullAccess + CloudWatchLogsFullAccess`.

---

### 2. OpenAI API Key (for LLM-powered RAG synthesis)

| Credential | Environment variable set at deploy time |
|---|---|
| OpenAI API key | `MSR_OPENAI_API_KEY` |

This key is passed to the Lambda function as an encrypted environment variable
(`NoEcho: true` in `template.yaml`).  It enables:
- Embedding via `text-embedding-3-small`
- Answer synthesis via `gpt-4o-mini`

**Without this key** the service falls back to random-projection embeddings and
returns only the retrieved text chunks without LLM synthesis.  The demo
notebook still works but answers are less coherent.

Estimated cost for the demo notebook (≈20 queries, ≈10 ingestion calls):
< $0.05 at current OpenAI pricing.

---

### 3. MSR API Key (self-generated; protects the deployed endpoint)

| Credential | Environment variable set at deploy time |
|---|---|
| Any secret string you choose | `MSR_API_KEY` |

Example: `openssl rand -hex 24`

This value is shared with the paper authors so they can authenticate their
notebook calls.  Leave empty to deploy without authentication (acceptable for
a short-lived demo; not recommended if the URL is widely shared).

---

### 4. Optional (improves rate limits, not required)

| Credential | Variable | Purpose |
|---|---|---|
| Your email address | `MSR_OPENALEX_EMAIL` | OpenAlex polite-pool (higher rate limit for literature ingestion) |
| GitHub Personal Access Token | `MSR_GITHUB_TOKEN` | Higher rate limit for fetching ORNL archive from `pranavkantgaur/msr-archive` |

---

## Deployment Steps (once credentials are ready)

```bash
# 1. Install the AWS SAM CLI (one-time)
pip install aws-sam-cli   # or: brew install aws-sam-cli

# 2. Configure AWS credentials
aws configure
#   AWS Access Key ID     [None]: <your-access-key-id>
#   AWS Secret Access Key [None]: <your-secret-access-key>
#   Default region name   [None]: us-east-1
#   Default output format [None]: json

# 3. Set credentials in your shell environment
export MSR_OPENAI_API_KEY="sk-..."
export MSR_API_KEY="$(openssl rand -hex 24)"   # save this value – share it with the authors
export MSR_OPENALEX_EMAIL="your@email.com"     # optional
export MSR_GITHUB_TOKEN="ghp_..."              # optional

# 4. Build + deploy (interactive first time; saves config to samconfig.toml)
make deploy-guided
#   ↑ SAM will ask you to confirm the parameter values and stack name.
#   Accept the defaults.  The deploy takes 3–5 minutes.

# 5. Get the deployed API URL
make outputs
#   Copy the 'ApiBaseUrl' value, e.g.:
#   https://abc123xyz.execute-api.us-east-1.amazonaws.com/prod

# 6. Seed the knowledge base with the ORNL archive + OpenAlex literature
API_URL=$(aws cloudformation describe-stacks \
  --stack-name msr-kb-service \
  --query "Stacks[0].Outputs[?OutputKey=='ApiBaseUrl'].OutputValue" \
  --output text)

curl -s -X POST "$API_URL/kb/update" \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: $MSR_API_KEY" \
  -d '{"source": "all"}' | python3 -m json.tool

# 7. Verify
curl -s "$API_URL/health" | python3 -m json.tool

# 8. Share with the paper authors:
#   - API_BASE_URL: the value from step 5
#   - API_KEY: the MSR_API_KEY value from step 3
#   - Notebook: use_cases/lucas_et_al_2025_demo.ipynb
```

---

## Cost Estimate (AWS, demo period of ~7 days)

| Service | Estimate |
|---|---|
| Lambda invocations (≈1 000 calls) | < $0.01 |
| API Gateway (≈1 000 requests) | < $0.01 |
| S3 storage (KB files ≈ 50 MB) | < $0.01 |
| CloudWatch Logs (1 week) | < $0.01 |
| **Total** | **< $0.05** |

The demo is essentially free on AWS for short-term use.

---

## Teardown (after the demo)

```bash
make delete   # removes the CloudFormation stack and all resources
```

> **Note:** The S3 bucket (KB storage) will be deleted along with the stack.
> Download a backup first if you want to preserve the ingested data:
> ```bash
> aws s3 sync s3://msr-kb-store-<account-id>-prod/ ./kb_backup/
> ```
