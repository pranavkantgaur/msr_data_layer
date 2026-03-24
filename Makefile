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
#
# GPU container (for GPU-accelerated embeddings + generation)
# -----------------------------------------------------------
#   make build-gpu-container       # build the GPU Docker image locally
#   make run-gpu-local             # run GPU container with CPU/GPU locally
#   make push-gpu-container        # push to ECR (set ECR_REPO first)
#   make deploy-gpu                # deploy GPU Lambda variant

# ---------------------------------------------------------------------------
# Configuration – override on the command line or in your environment
# ---------------------------------------------------------------------------

STACK_NAME   ?= msr-kb-service
REGION       ?= us-east-1
ENVIRONMENT  ?= prod
S3_DEPLOY_BUCKET ?= # SAM deployment artifacts bucket (created on first deploy)

# GPU container settings
GPU_IMAGE_NAME  ?= msr-kb-gpu
GPU_IMAGE_TAG   ?= latest
ECR_REPO        ?= # e.g. 123456789012.dkr.ecr.us-east-1.amazonaws.com/msr-kb-gpu
EMBED_MODEL     ?= sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL       ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo ""
	@echo "MSR Data Layer – make targets"
	@echo "============================="
	@echo ""
	@echo "─── Primary: GitHub Codespaces / local ──────────────────────────────"
	@echo "  make serve              Start HTTP server on port 8000 (server.py)"
	@echo "                          In Codespaces the URL is auto-published."
	@echo "  make serve-mcp          Start stdio MCP server (for AI agents)"
	@echo "  make health             GET  http://127.0.0.1:8000/health"
	@echo "  make query              POST http://127.0.0.1:8000/query (test query)"
	@echo "  make test               Run all unit tests"
	@echo ""
	@echo "─── Advanced: AWS Lambda deployment (optional) ──────────────────────"
	@echo "  make build              Build Lambda deployment package (Docker)"
	@echo "  make build-native       Build without Docker (uses current Python)"
	@echo "  make deploy-guided      First-time interactive deploy"
	@echo "  make deploy             Deploy using saved samconfig.toml"
	@echo "  make delete             Delete the CloudFormation stack"
	@echo "  make local-api          Start SAM local HTTP server (port 3000)"
	@echo "  make local-health       GET  http://127.0.0.1:3000/health"
	@echo "  make local-query        POST http://127.0.0.1:3000/query"
	@echo "  make local-update       POST http://127.0.0.1:3000/kb/update"
	@echo "  make logs               Tail Lambda CloudWatch logs"
	@echo "  make outputs            Show CloudFormation stack outputs (API URL)"
	@echo ""
	@echo "─── Advanced: GPU container (optional) ─────────────────────────────"
	@echo "  make build-gpu-container  Build GPU Docker image (Dockerfile.gpu)"
	@echo "  make run-gpu-local        Run GPU container locally (CPU fallback)"
	@echo "  make run-gpu-cuda         Run GPU container with NVIDIA GPU"
	@echo "  make push-gpu-container   Push GPU image to ECR"
	@echo "  make deploy-gpu           Deploy GPU Lambda variant via SAM"
	@echo ""
	@echo "Configuration:"
	@echo "  STACK_NAME=$(STACK_NAME)"
	@echo "  REGION=$(REGION)"
	@echo "  ENVIRONMENT=$(ENVIRONMENT)"
	@echo "  ECR_REPO=$(ECR_REPO)"
	@echo "  EMBED_MODEL=$(EMBED_MODEL)"
	@echo "  LLM_MODEL=$(LLM_MODEL)"
	@echo ""
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
# Primary targets – GitHub Codespaces / local server (no AWS required)
# ---------------------------------------------------------------------------

.PHONY: serve
serve:
	python server.py --host 0.0.0.0 --port $(MSR_SERVER_PORT)

MSR_SERVER_PORT ?= 8000

.PHONY: serve-mcp
serve-mcp:
	python msr_mcp_server_main.py

.PHONY: health
health:
	curl -s http://127.0.0.1:$(MSR_SERVER_PORT)/health | python3 -m json.tool

.PHONY: query
query:
	curl -s -X POST http://127.0.0.1:$(MSR_SERVER_PORT)/query \
		-H "Content-Type: application/json" \
		-d '{"question": "What is the thermal efficiency of the TMSR-LF1 reactor?"}' \
		| python3 -m json.tool

# ---------------------------------------------------------------------------
# Local development (requires sam local – optional, AWS path)
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

# GPU remote targets (hit /gpu/* routes)
.PHONY: remote-gpu-health
remote-gpu-health:
	curl -s "$(_API_URL)/gpu/health" | python3 -m json.tool

.PHONY: remote-gpu-query
remote-gpu-query:
	@test -n "$(QUESTION)" || (echo "Usage: make remote-gpu-query QUESTION='your question'" && exit 1)
	curl -s -X POST "$(_API_URL)/gpu/query" \
		-H "Content-Type: application/json" \
		-H "X-Api-Key: $${MSR_API_KEY:-}" \
		-d "{\"question\": \"$(QUESTION)\"}" \
		| python3 -m json.tool

# ---------------------------------------------------------------------------
# GPU container targets
# ---------------------------------------------------------------------------

# Build the GPU-capable Docker image using Dockerfile.gpu
.PHONY: build-gpu-container
build-gpu-container:
	docker build \
		-f Dockerfile.gpu \
		--build-arg EMBED_MODEL="$(EMBED_MODEL)" \
		--build-arg LLM_MODEL="$(LLM_MODEL)" \
		-t $(GPU_IMAGE_NAME):$(GPU_IMAGE_TAG) \
		.

# Run the GPU container locally on CPU (no --gpus flag)
.PHONY: run-gpu-local
run-gpu-local:
	docker run --rm \
		-p 9000:8080 \
		-e MSR_USE_LOCAL_GPU=true \
		-e MSR_API_KEY=$${MSR_API_KEY:-} \
		-e MSR_OPENAI_API_KEY="" \
		$(GPU_IMAGE_NAME):$(GPU_IMAGE_TAG)

# Run the GPU container with NVIDIA GPU (requires nvidia-container-toolkit)
.PHONY: run-gpu-cuda
run-gpu-cuda:
	docker run --rm --gpus all \
		-p 9000:8080 \
		-e MSR_USE_LOCAL_GPU=true \
		-e MSR_API_KEY=$${MSR_API_KEY:-} \
		$(GPU_IMAGE_NAME):$(GPU_IMAGE_TAG)

# Test the locally running GPU container via Lambda RIE
.PHONY: test-gpu-local-health
test-gpu-local-health:
	curl -s -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
		-d '{"rawPath":"/health","requestContext":{"http":{"method":"GET"}},"headers":{}}' \
		| python3 -m json.tool

.PHONY: test-gpu-local-query
test-gpu-local-query:
	@test -n "$(QUESTION)" || (echo "Usage: make test-gpu-local-query QUESTION='your question'" && exit 1)
	curl -s -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" \
		-d "{\"rawPath\":\"/query\",\"requestContext\":{\"http\":{\"method\":\"POST\"}},\"body\":\"{\\\"question\\\":\\\"$(QUESTION)\\\"}\"}" \
		| python3 -m json.tool

# Authenticate with ECR and push the GPU image
.PHONY: push-gpu-container
push-gpu-container:
	@test -n "$(ECR_REPO)" || (echo "Error: ECR_REPO is not set. Usage: make push-gpu-container ECR_REPO=<ecr-uri>" && exit 1)
	aws ecr get-login-password --region $(REGION) \
		| docker login --username AWS --password-stdin $$(echo $(ECR_REPO) | cut -d/ -f1)
	docker tag $(GPU_IMAGE_NAME):$(GPU_IMAGE_TAG) $(ECR_REPO):$(GPU_IMAGE_TAG)
	docker push $(ECR_REPO):$(GPU_IMAGE_TAG)

# Create the ECR repository (idempotent)
.PHONY: create-ecr-repo
create-ecr-repo:
	aws ecr create-repository \
		--repository-name msr-kb-gpu \
		--region $(REGION) \
		--image-scanning-configuration scanOnPush=true \
		--encryption-configuration encryptionType=AES256 \
		2>/dev/null || echo "Repository already exists."

# Deploy the GPU Lambda variant via SAM (requires image already pushed to ECR)
.PHONY: deploy-gpu
deploy-gpu: build
	@test -n "$(ECR_REPO)" || (echo "Error: ECR_REPO is not set. Usage: make deploy-gpu ECR_REPO=<ecr-uri>" && exit 1)
	sam deploy \
		--stack-name $(STACK_NAME) \
		--region $(REGION) \
		--capabilities CAPABILITY_IAM \
		--no-confirm-changeset \
		--parameter-overrides \
			UseLocalGPU=true \
			GPUContainerImageUri=$(ECR_REPO):$(GPU_IMAGE_TAG) \
			LocalEmbedModel="$(EMBED_MODEL)" \
			LocalLLMModel="$(LLM_MODEL)" \
			MsrApiKey=$${MSR_API_KEY:-}

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

.PHONY: test
test:
	python -m pytest test_lambda_function.py test_msr_mcp_server.py test_msr_rag.py test_msr_kb_sources.py test_generate_architecture_diagrams.py -v

.PHONY: test-lambda
test-lambda:
	python -m pytest test_lambda_function.py -v

# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------

.PHONY: logs
logs:
	sam logs -n MSRKBFunction --stack-name $(STACK_NAME) --region $(REGION) --tail

.PHONY: logs-gpu
logs-gpu:
	sam logs -n MSRKBGPUFunction --stack-name $(STACK_NAME) --region $(REGION) --tail
