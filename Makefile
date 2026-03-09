# MSR Knowledge Base Service – Build and Deploy Makefile
#
# Prerequisites
# -------------
#   pip install aws-sam-cli        # SAM CLI  https://docs.aws.amazon.com/serverless-application-model/
#   aws configure                  # AWS credentials
#
# Quick start
# -----------
#   make build
#   make deploy-guided             # interactive first-time deploy (creates samconfig.toml)
#   make deploy                    # subsequent deploys using saved config
#
# Local development
# -----------------
#   make local-api                 # start local HTTP server on http://127.0.0.1:3000
#   make local-health              # call the local /health endpoint
#   make local-query               # send a test query to the local /query endpoint

# ---------------------------------------------------------------------------
# Configuration – override on the command line or in your environment
# ---------------------------------------------------------------------------

STACK_NAME   ?= msr-kb-service
REGION       ?= us-east-1
ENVIRONMENT  ?= prod
S3_DEPLOY_BUCKET ?= # SAM deployment artifacts bucket (created on first deploy)

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo ""
	@echo "MSR Knowledge Base Service – make targets"
	@echo "========================================="
	@echo ""
	@echo "  make build          Build Lambda deployment package (Docker)"
	@echo "  make build-native   Build without Docker (uses current Python)"
	@echo "  make deploy-guided  First-time interactive deploy"
	@echo "  make deploy         Deploy using saved samconfig.toml"
	@echo "  make delete         Delete the CloudFormation stack"
	@echo ""
	@echo "  make local-api      Start local HTTP server (port 3000)"
	@echo "  make local-health   GET  http://127.0.0.1:3000/health"
	@echo "  make local-query    POST http://127.0.0.1:3000/query"
	@echo "  make local-update   POST http://127.0.0.1:3000/kb/update"
	@echo ""
	@echo "  make test           Run unit tests"
	@echo "  make logs           Tail Lambda CloudWatch logs"
	@echo "  make outputs        Show CloudFormation stack outputs (API URL etc.)"
	@echo ""
	@echo "Configuration:"
	@echo "  STACK_NAME=$(STACK_NAME)"
	@echo "  REGION=$(REGION)"
	@echo "  ENVIRONMENT=$(ENVIRONMENT)"
	@echo ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

# Build inside a Lambda-compatible Docker container (recommended – ensures
# native numpy binaries match the Amazon Linux 2023 runtime).
.PHONY: build
build:
	sam build \
		--use-container \
		--build-image public.ecr.aws/sam/build-python3.12:latest \
		--template template.yaml

# Build without Docker (faster, but native binaries must match the host).
# Use only if running on Amazon Linux 2023 or for quick local iteration.
.PHONY: build-native
build-native:
	sam build --template template.yaml

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

# First-time deploy – walks through all parameters interactively and saves
# the config to samconfig.toml for subsequent runs.
.PHONY: deploy-guided
deploy-guided: build
	sam deploy \
		--guided \
		--stack-name $(STACK_NAME) \
		--region $(REGION) \
		--capabilities CAPABILITY_IAM \
		--parameter-overrides \
			Environment=$(ENVIRONMENT) \
			OpenAIApiKey=$${MSR_OPENAI_API_KEY:-} \
			MsrApiKey=$${MSR_API_KEY:-} \
			OpenAlexEmail=$${MSR_OPENALEX_EMAIL:-} \
			GithubToken=$${MSR_GITHUB_TOKEN:-}

# Subsequent deploys using saved samconfig.toml.
.PHONY: deploy
deploy: build
	sam deploy \
		--stack-name $(STACK_NAME) \
		--region $(REGION) \
		--capabilities CAPABILITY_IAM \
		--no-confirm-changeset

# ---------------------------------------------------------------------------
# Stack management
# ---------------------------------------------------------------------------

.PHONY: delete
delete:
	@echo "WARNING: This will delete all resources including the KB S3 bucket."
	@read -p "Type 'yes' to confirm: " ans && [ "$$ans" = "yes" ]
	sam delete --stack-name $(STACK_NAME) --region $(REGION)

.PHONY: outputs
outputs:
	@aws cloudformation describe-stacks \
		--stack-name $(STACK_NAME) \
		--region $(REGION) \
		--query 'Stacks[0].Outputs' \
		--output table

# ---------------------------------------------------------------------------
# Local development (requires sam local)
# ---------------------------------------------------------------------------

.PHONY: local-api
local-api: build-native
	sam local start-api \
		--template template.yaml \
		--port 3000 \
		--env-vars env.json 2>/dev/null || \
	sam local start-api \
		--template template.yaml \
		--port 3000

.PHONY: local-health
local-health:
	curl -s http://127.0.0.1:3000/health | python3 -m json.tool

.PHONY: local-query
local-query:
	curl -s -X POST http://127.0.0.1:3000/query \
		-H "Content-Type: application/json" \
		-d '{"question": "What is the thermal efficiency of the TMSR-LF1 reactor?"}' \
		| python3 -m json.tool

.PHONY: local-mcp
local-mcp:
	curl -s -X POST http://127.0.0.1:3000/mcp \
		-H "Content-Type: application/json" \
		-d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
		| python3 -m json.tool

.PHONY: local-update
local-update:
	curl -s -X POST http://127.0.0.1:3000/kb/update \
		-H "Content-Type: application/json" \
		-d '{"source": "archive", "max_docs": 5}' \
		| python3 -m json.tool

# ---------------------------------------------------------------------------
# Remote convenience targets (uses STACK_NAME stack outputs)
# ---------------------------------------------------------------------------

# Retrieve the deployed API base URL from CloudFormation outputs
_API_URL = $(shell aws cloudformation describe-stacks \
	--stack-name $(STACK_NAME) --region $(REGION) \
	--query "Stacks[0].Outputs[?OutputKey=='ApiBaseUrl'].OutputValue" \
	--output text 2>/dev/null)

.PHONY: remote-health
remote-health:
	curl -s "$(_API_URL)/health" | python3 -m json.tool

.PHONY: remote-query
remote-query:
	@test -n "$(QUESTION)" || (echo "Usage: make remote-query QUESTION='your question'" && exit 1)
	curl -s -X POST "$(_API_URL)/query" \
		-H "Content-Type: application/json" \
		-H "X-Api-Key: $${MSR_API_KEY:-}" \
		-d "{\"question\": \"$(QUESTION)\"}" \
		| python3 -m json.tool

.PHONY: remote-update
remote-update:
	curl -s -X POST "$(_API_URL)/kb/update" \
		-H "Content-Type: application/json" \
		-H "X-Api-Key: $${MSR_API_KEY:-}" \
		-d '{"source": "all"}' \
		| python3 -m json.tool

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

.PHONY: test
test:
	python -m pytest test_lambda_function.py test_msr_mcp_server.py test_msr_rag.py test_msr_kb_sources.py -v

.PHONY: test-lambda
test-lambda:
	python -m pytest test_lambda_function.py -v

# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

.PHONY: logs
logs:
	sam logs -n MSRKBFunction --stack-name $(STACK_NAME) --region $(REGION) --tail
