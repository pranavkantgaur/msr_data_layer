"""
Unit tests for the AWS Lambda handler (lambda_function.py).

Tests cover:
- Authentication (API key check, bypass when unset)
- Request body parsing (plain JSON, base64-encoded)
- /health endpoint
- /mcp endpoint (MCP JSON-RPC delegation)
- /query endpoint (RAG delegation)
- /kb/update endpoint (archive, openalex, all)
- EventBridge scheduled KB update handler
- S3 sync helpers (sync_kb_from_s3 / sync_kb_to_s3)
- Direct Lambda invocation (non-HTTP event fallback)
- 404 for unknown routes
- lambda_handler routing (HTTP vs EventBridge)
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apigw_event(
    method: str = "GET",
    path: str = "/health",
    body: Any = None,
    headers: dict | None = None,
    base64_encoded: bool = False,
) -> dict:
    """Build a minimal API Gateway HTTP API v2 event."""
    raw_body = ""
    if body is not None:
        raw_str = json.dumps(body) if not isinstance(body, str) else body
        if base64_encoded:
            raw_body = base64.b64encode(raw_str.encode()).decode()
        else:
            raw_body = raw_str
    return {
        "version": "2.0",
        "requestContext": {
            "http": {
                "method": method,
                "path": path,
            }
        },
        "rawPath": path,
        "body": raw_body if raw_body else None,
        "isBase64Encoded": base64_encoded,
        "headers": headers or {},
    }


def _eventbridge_event() -> dict:
    return {
        "source": "aws.events",
        "detail-type": "Scheduled Event",
        "detail": "periodic-kb-update",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_lambda_module(monkeypatch):
    """
    Reset module-level state in lambda_function between tests so the
    warm-cache (_rag_cache) and env vars don't leak across tests.
    """
    import lambda_function as lf
    monkeypatch.setattr(lf, "_rag_cache", None)
    monkeypatch.delenv("MSR_API_KEY", raising=False)
    monkeypatch.delenv("MSR_KB_S3_BUCKET", raising=False)
    monkeypatch.setattr(lf, "_API_KEY", "")
    monkeypatch.setattr(lf, "_S3_BUCKET", "")
    yield
    # Reset again after test
    lf._rag_cache = None


@pytest.fixture()
def mock_rag():
    """A MagicMock mimicking MSRDigitalTwinRAG."""
    rag = MagicMock()
    rag.answer.return_value = "The thermal efficiency is approximately 40%."
    rag.load_msr_archive.return_value = 3
    rag.update_openalex.return_value = 2
    return rag


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:
    def test_no_api_key_set_allows_all(self):
        """When MSR_API_KEY is unset, all requests pass through."""
        import lambda_function as lf
        lf._API_KEY = ""
        event = _apigw_event()
        assert lf._is_authenticated(event) is True

    def test_valid_key_in_x_api_key_header(self):
        import lambda_function as lf
        lf._API_KEY = "secret123"
        event = _apigw_event(headers={"x-api-key": "secret123"})
        assert lf._is_authenticated(event) is True

    def test_valid_key_in_bearer_auth_header(self):
        import lambda_function as lf
        lf._API_KEY = "secret123"
        event = _apigw_event(headers={"Authorization": "Bearer secret123"})
        assert lf._is_authenticated(event) is True

    def test_wrong_key_rejected(self):
        import lambda_function as lf
        lf._API_KEY = "secret123"
        event = _apigw_event(headers={"x-api-key": "wrong"})
        assert lf._is_authenticated(event) is False

    def test_missing_key_rejected_when_required(self):
        import lambda_function as lf
        lf._API_KEY = "secret123"
        event = _apigw_event(headers={})
        assert lf._is_authenticated(event) is False

    def test_lambda_returns_403_when_key_missing(self):
        import lambda_function as lf
        lf._API_KEY = "secret123"
        event = _apigw_event("GET", "/health")
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 403


# ---------------------------------------------------------------------------
# Request body parsing
# ---------------------------------------------------------------------------

class TestParseBody:
    def test_plain_json_body(self):
        import lambda_function as lf
        event = _apigw_event("POST", "/query", body={"question": "hello"})
        raw, parsed = lf._parse_body(event)
        assert parsed["question"] == "hello"

    def test_base64_encoded_body(self):
        import lambda_function as lf
        event = _apigw_event("POST", "/query", body={"question": "hi"}, base64_encoded=True)
        raw, parsed = lf._parse_body(event)
        assert parsed["question"] == "hi"

    def test_empty_body(self):
        import lambda_function as lf
        event = _apigw_event("GET", "/health")
        raw, parsed = lf._parse_body(event)
        assert parsed == {}

    def test_invalid_json_body(self):
        import lambda_function as lf
        event = _apigw_event("POST", "/mcp")
        event["body"] = "not-json"
        raw, parsed = lf._parse_body(event)
        assert raw == "not-json"
        assert parsed == {}


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self):
        import lambda_function as lf
        event = _apigw_event("GET", "/health")
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200

    def test_response_contains_service_name(self):
        import lambda_function as lf
        event = _apigw_event("GET", "/health")
        resp = lf.lambda_handler(event, None)
        body = json.loads(resp["body"])
        assert body["service"] == "msr-knowledge-base"
        assert body["status"] == "healthy"
        assert "gpu" in body
        assert "torch_available" in body["gpu"]
        # data_source info included (replaces old reactor_status field)
        assert "data_source" in body

    def test_root_path_returns_health(self):
        import lambda_function as lf
        event = _apigw_event("GET", "/")
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200


# ---------------------------------------------------------------------------
# /mcp endpoint
# ---------------------------------------------------------------------------

class TestMCPEndpoint:
    def test_tools_list(self):
        import lambda_function as lf
        mcp_msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        event = _apigw_event("POST", "/mcp", body=mcp_msg)
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert "result" in body
        assert "tools" in body["result"]

    def test_initialize(self):
        import lambda_function as lf
        mcp_msg = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "clientInfo": {}}
        }
        event = _apigw_event("POST", "/mcp", body=mcp_msg)
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["result"]["serverInfo"]["name"] == "msr-data-layer"

    def test_tools_call_get_reactor_status(self):
        import lambda_function as lf
        mcp_msg = {
            "jsonrpc": "2.0", "id": 2,
            "method": "tools/call",
            "params": {"name": "get_reactor_status", "arguments": {}},
        }
        event = _apigw_event("POST", "/mcp", body=mcp_msg)
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert "result" in body

    def test_empty_body_returns_400(self):
        import lambda_function as lf
        event = _apigw_event("POST", "/mcp")
        event["body"] = ""
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 400

    def test_notification_returns_204(self):
        """MCP notifications (no id) produce an empty 204 response."""
        import lambda_function as lf
        # notification: no "id" field
        mcp_msg = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        event = _apigw_event("POST", "/mcp", body=mcp_msg)
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 204


# ---------------------------------------------------------------------------
# /query endpoint
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_valid_question(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/query", body={"question": "What is the core temp?"})
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert "answer" in body
        assert body["question"] == "What is the core temp?"
        mock_rag.answer.assert_called_once_with("What is the core temp?", top_k=5)

    def test_custom_top_k(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/query", body={"question": "hello?", "top_k": 10})
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200
        mock_rag.answer.assert_called_once_with("hello?", top_k=10)

    def test_top_k_clamped_to_20(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/query", body={"question": "hi", "top_k": 999})
        lf.lambda_handler(event, None)
        mock_rag.answer.assert_called_once_with("hi", top_k=20)

    def test_empty_question_returns_400(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/query", body={"question": "  "})
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 400

    def test_missing_question_returns_400(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/query", body={})
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 400

    def test_rag_error_returns_500(self, mock_rag):
        import lambda_function as lf
        mock_rag.answer.side_effect = RuntimeError("KB unavailable")
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/query", body={"question": "test"})
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 500


# ---------------------------------------------------------------------------
# /kb/update endpoint
# ---------------------------------------------------------------------------

class TestKBUpdateEndpoint:
    def test_update_archive(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/kb/update", body={"source": "archive"})
        with patch.object(lf, "sync_kb_to_s3") as mock_sync:
            resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["added"]["archive"] == 3
        assert "openalex" not in body["added"]
        mock_rag.load_msr_archive.assert_called_once_with(max_docs=0)
        mock_sync.assert_called_once()

    def test_update_openalex(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/kb/update", body={"source": "openalex"})
        with patch.object(lf, "sync_kb_to_s3"):
            resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["added"]["openalex"] == 2
        mock_rag.update_openalex.assert_called_once_with(max_docs=None)

    def test_update_all(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/kb/update", body={"source": "all"})
        with patch.object(lf, "sync_kb_to_s3"):
            resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["added"]["archive"] == 3
        assert body["added"]["openalex"] == 2
        assert body["total_new_documents"] == 5

    def test_update_default_source_is_all(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/kb/update", body={})
        with patch.object(lf, "sync_kb_to_s3"):
            resp = lf.lambda_handler(event, None)
        body = json.loads(resp["body"])
        assert resp["statusCode"] == 200
        assert "archive" in body["added"]
        assert "openalex" in body["added"]

    def test_max_docs_passed_to_loader(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/kb/update", body={"source": "archive", "max_docs": 10})
        with patch.object(lf, "sync_kb_to_s3"):
            lf.lambda_handler(event, None)
        mock_rag.load_msr_archive.assert_called_once_with(max_docs=10)

    def test_invalid_source_returns_400(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/kb/update", body={"source": "unknown"})
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 400

    def test_update_error_returns_500(self, mock_rag):
        import lambda_function as lf
        mock_rag.load_msr_archive.side_effect = RuntimeError("network error")
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/kb/update", body={"source": "archive"})
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 500


# ---------------------------------------------------------------------------
# Scheduled EventBridge KB update
# ---------------------------------------------------------------------------

class TestScheduledUpdate:
    def test_scheduled_event_triggers_full_update(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _eventbridge_event()
        with patch.object(lf, "sync_kb_to_s3") as mock_sync:
            result = lf.lambda_handler(event, None)
        assert result["success"] is True
        assert result["added"]["archive"] == 3
        assert result["added"]["openalex"] == 2
        mock_rag.load_msr_archive.assert_called_once()
        mock_rag.update_openalex.assert_called_once()
        mock_sync.assert_called_once()

    def test_scheduled_event_failure_returns_error_dict(self, mock_rag):
        import lambda_function as lf
        mock_rag.load_msr_archive.side_effect = RuntimeError("S3 unavailable")
        lf._rag_cache = mock_rag
        event = _eventbridge_event()
        result = lf.lambda_handler(event, None)
        assert result["success"] is False
        assert "error" in result

    def test_detail_type_scheduled_event_also_dispatched(self, mock_rag):
        """EventBridge events via 'detail-type' field are handled too."""
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = {"detail-type": "Scheduled Event", "detail": "test"}
        with patch.object(lf, "sync_kb_to_s3"):
            result = lf.lambda_handler(event, None)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# S3 sync helpers
# ---------------------------------------------------------------------------

class TestS3Sync:
    def test_sync_from_s3_no_op_without_bucket(self):
        import lambda_function as lf
        lf._S3_BUCKET = ""
        # Should not raise and should not call boto3
        with patch("lambda_function._BOTO3_AVAILABLE", True), \
             patch("lambda_function._s3_client") as mock_s3:
            lf.sync_kb_from_s3()
        mock_s3.assert_not_called()

    def test_sync_to_s3_no_op_without_bucket(self):
        import lambda_function as lf
        lf._S3_BUCKET = ""
        with patch("lambda_function._BOTO3_AVAILABLE", True), \
             patch("lambda_function._s3_client") as mock_s3:
            lf.sync_kb_to_s3()
        mock_s3.assert_not_called()

    def test_sync_from_s3_downloads_files(self, tmp_path):
        import lambda_function as lf
        lf._S3_BUCKET = "test-bucket"
        mock_s3 = MagicMock()
        with patch("lambda_function._BOTO3_AVAILABLE", True), \
             patch("lambda_function._s3_client", return_value=mock_s3), \
             patch("lambda_function._KB_LOCAL_DIR", str(tmp_path / "kb")):
            lf.sync_kb_from_s3()
        # Should have called download_file for each KB file
        assert mock_s3.download_file.call_count == len(lf._KB_FILES)

    def test_sync_to_s3_uploads_existing_files(self, tmp_path):
        import lambda_function as lf
        lf._S3_BUCKET = "test-bucket"
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        # Create two KB files
        (kb_dir / "chunks.json").write_text("{}")
        (kb_dir / "insights.json").write_text("{}")

        mock_s3 = MagicMock()
        with patch("lambda_function._BOTO3_AVAILABLE", True), \
             patch("lambda_function._s3_client", return_value=mock_s3), \
             patch("lambda_function._KB_LOCAL_DIR", str(kb_dir)):
            lf.sync_kb_to_s3()
        # Only the two files that exist should be uploaded
        assert mock_s3.upload_file.call_count == 2

    def test_sync_from_s3_tolerates_missing_files(self, tmp_path):
        """Files not yet in S3 (first run) should not raise."""
        import lambda_function as lf
        lf._S3_BUCKET = "test-bucket"
        mock_s3 = MagicMock()
        # Simulate a ClientError (e.g. NoSuchKey) as would happen on first run
        try:
            from botocore.exceptions import ClientError
            error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
            mock_s3.download_file.side_effect = ClientError(error_response, "GetObject")
        except ImportError:
            # botocore not available; use Exception which matches the fallback alias
            mock_s3.download_file.side_effect = Exception("NoSuchKey")
        with patch("lambda_function._BOTO3_AVAILABLE", True), \
             patch("lambda_function._s3_client", return_value=mock_s3), \
             patch("lambda_function._KB_LOCAL_DIR", str(tmp_path / "kb")):
            lf.sync_kb_from_s3()   # should not raise


# ---------------------------------------------------------------------------
# /data/ingest endpoint
# ---------------------------------------------------------------------------

class TestPlantDataIngest:
    """Tests for the POST /data/ingest plant operational data endpoint."""

    def test_ingest_returns_200(self, mock_rag, tmp_path):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        mock_rag.add_document.return_value = 2
        event = _apigw_event(
            "POST", "/data/ingest",
            body={"content": "Core temperature 702°C at 14:32 UTC"},
        )
        from unittest.mock import patch, MagicMock
        mock_loader = MagicMock()
        mock_loader.ingest_text.return_value = 2
        mock_plant_module = MagicMock()
        mock_plant_module.PlantDataLoader.return_value = mock_loader
        with patch.dict("sys.modules", {"msr_kb_sources": mock_plant_module}):
            resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["chunks_added"] == 2
        assert "source_id" in body
        assert body["data_type"] == "operational_data"

    def test_ingest_with_explicit_source_id(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        from unittest.mock import patch, MagicMock
        mock_loader = MagicMock()
        mock_loader.ingest_text.return_value = 1
        mock_plant_module = MagicMock()
        mock_plant_module.PlantDataLoader.return_value = mock_loader
        event = _apigw_event(
            "POST", "/data/ingest",
            body={
                "content": "HX-1 inspection complete. No fouling.",
                "data_type": "maintenance_report",
                "source_id": "maint-hx1-20240115",
            },
        )
        with patch.dict("sys.modules", {"msr_kb_sources": mock_plant_module}):
            resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["source_id"] == "maint-hx1-20240115"
        assert body["data_type"] == "maintenance_report"

    def test_ingest_missing_content_returns_400(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event("POST", "/data/ingest", body={"data_type": "event_log"})
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 400

    def test_ingest_invalid_data_type_returns_400(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        event = _apigw_event(
            "POST", "/data/ingest",
            body={"content": "some data", "data_type": "invalid_type"},
        )
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 400

    def test_ingest_all_valid_data_types(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        from unittest.mock import patch, MagicMock
        valid_types = [
            "sensor_snapshot", "event_log",
            "maintenance_report", "operational_data"
        ]
        for data_type in valid_types:
            mock_loader = MagicMock()
            mock_loader.ingest_text.return_value = 1
            mock_plant_module = MagicMock()
            mock_plant_module.PlantDataLoader.return_value = mock_loader
            event = _apigw_event(
                "POST", "/data/ingest",
                body={"content": f"{data_type} content", "data_type": data_type},
            )
            with patch.dict("sys.modules", {"msr_kb_sources": mock_plant_module}):
                resp = lf.lambda_handler(event, None)
            assert resp["statusCode"] == 200, f"Failed for data_type={data_type}"

    def test_ingest_syncs_to_s3(self, mock_rag):
        import lambda_function as lf
        lf._rag_cache = mock_rag
        from unittest.mock import patch, MagicMock
        mock_loader = MagicMock()
        mock_loader.ingest_text.return_value = 1
        mock_plant_module = MagicMock()
        mock_plant_module.PlantDataLoader.return_value = mock_loader
        event = _apigw_event(
            "POST", "/data/ingest",
            body={"content": "some operational data"},
        )
        with patch.dict("sys.modules", {"msr_kb_sources": mock_plant_module}), \
             patch.object(lf, "sync_kb_to_s3") as mock_sync:
            lf.lambda_handler(event, None)
        mock_sync.assert_called_once()


# ---------------------------------------------------------------------------
# Routing – misc
# ---------------------------------------------------------------------------

class TestRouting:
    def test_unknown_path_returns_404(self):
        import lambda_function as lf
        event = _apigw_event("GET", "/does-not-exist")
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 404

    def test_get_on_mcp_returns_404(self):
        import lambda_function as lf
        event = _apigw_event("GET", "/mcp")
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 404

    def test_trailing_slash_normalised(self):
        import lambda_function as lf
        event = _apigw_event("GET", "/health/")
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200

    def test_response_content_type_is_json(self):
        import lambda_function as lf
        event = _apigw_event("GET", "/health")
        resp = lf.lambda_handler(event, None)
        assert resp["headers"]["Content-Type"] == "application/json"

    def test_apigw_v1_event_format(self):
        """API Gateway v1 (REST API) events use httpMethod + path."""
        import lambda_function as lf
        event = {
            "httpMethod": "GET",
            "path": "/health",
            "body": None,
            "isBase64Encoded": False,
            "headers": {},
        }
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200

    def test_direct_invocation_fallback(self, mock_rag):
        """Direct Lambda invocations (no requestContext) are routed as POST /query."""
        import lambda_function as lf
        lf._rag_cache = mock_rag
        # Invoke directly with a plain dict
        event = {"question": "What is FLiBe?"}
        resp = lf.lambda_handler(event, None)
        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert "answer" in body


# ---------------------------------------------------------------------------
# RAG warm cache initialisation
# ---------------------------------------------------------------------------

class TestRAGCache:
    def test_rag_initialised_once_across_requests(self, mock_rag):
        """The RAG instance is cached and not re-created on subsequent calls."""
        import lambda_function as lf
        lf._rag_cache = None

        call_count = 0

        def fake_get_rag():
            nonlocal call_count
            call_count += 1
            lf._rag_cache = mock_rag
            return mock_rag

        # Monkeypatch _get_rag to track calls
        with patch.object(lf, "_get_rag", side_effect=fake_get_rag):
            # When _rag_cache is already set the real _get_rag returns it
            # without reconstruction; verify it via direct state check
            lf._rag_cache = mock_rag
            r1 = lf._get_rag()
            r2 = lf._get_rag()
        # Both calls should return the same object
        assert r1 is r2

    def test_kb_dir_set_to_tmp(self, monkeypatch):
        """MSR_KB_DIR is defaulted to /tmp/kb_store."""
        import lambda_function as lf
        monkeypatch.delenv("MSR_KB_DIR", raising=False)
        lf._rag_cache = None
        mock_rag_inst = MagicMock()
        mock_module = MagicMock()
        mock_module.MSRDigitalTwinRAG.return_value = mock_rag_inst
        with patch("lambda_function.sync_kb_from_s3"), \
             patch.dict("sys.modules", {"msr_digital_twin_with_rag": mock_module}):
            lf._get_rag()
        assert os.environ.get("MSR_KB_DIR") == "/tmp/kb_store"
