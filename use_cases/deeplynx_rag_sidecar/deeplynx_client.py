"""DeepLynx REST API client.

Wraps the DeepLynx Nexus REST API for use by the RAG sidecar.
Falls back to a stub when DEEPLYNX_URL is not set, so unit tests
and offline demos require zero network access.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_DEEPLYNX_URL = os.environ.get("DEEPLYNX_URL", "")
_DEEPLYNX_API_KEY = os.environ.get("DEEPLYNX_API_KEY", "")
_DEEPLYNX_API_SECRET = os.environ.get("DEEPLYNX_API_SECRET", "")

# Stub data used when DEEPLYNX_URL is unset.
_STUB_RECORDS: list[dict[str, Any]] = [
    {
        "id": "stub-001",
        "type": "DesignDocument",
        "properties": {
            "title": "Primary Loop Piping Specification",
            "content": (
                "Piping material: Hastelloy-N. Operating temperature: 650 °C. "
                "Pressure rating: 1.5 bar. Weld procedure: WPS-47."
            ),
            "created_at": "2024-06-01T00:00:00Z",
        },
    },
    {
        "id": "stub-002",
        "type": "InspectionReport",
        "properties": {
            "title": "Salt Pump Bearing Inspection – 2024-Q3",
            "content": (
                "No abnormal wear detected on primary salt pump bearings. "
                "Vibration signature nominal. Next inspection: 2025-Q1."
            ),
            "created_at": "2024-09-15T00:00:00Z",
        },
    },
    {
        "id": "stub-003",
        "type": "MaintenanceRecord",
        "properties": {
            "title": "Heat Exchanger Tube Bundle Cleaning – 2024-11",
            "content": (
                "FLiBe deposits removed from 12 tubes in the primary heat "
                "exchanger. Post-cleaning heat transfer coefficient improved 8%. "
                "Tubes 7, 14 flagged for replacement at next outage."
            ),
            "created_at": "2024-11-03T00:00:00Z",
        },
    },
]

_STUB_TIMESERIES: list[dict[str, Any]] = [
    {"timestamp": "2024-01-15T08:00:00Z", "sensor": "core_temp_c", "value": 620.1},
    {"timestamp": "2024-01-15T09:00:00Z", "sensor": "core_temp_c", "value": 623.4},
    {"timestamp": "2024-01-15T10:00:00Z", "sensor": "core_temp_c", "value": 635.8},
]


def _make_request(
    path: str,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    token: str | None = None,
) -> Any:
    """Execute an HTTP request against the DeepLynx base URL."""
    url = _DEEPLYNX_URL.rstrip("/") + path
    data = json.dumps(body).encode() if body else None
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif _DEEPLYNX_API_KEY:
        headers["x-api-key"] = _DEEPLYNX_API_KEY
        headers["x-api-secret"] = _DEEPLYNX_API_SECRET

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        logger.error("DeepLynx request failed: %s %s → %s", method, path, exc)
        return None


def _get_bearer_token() -> str | None:
    """Exchange API key/secret for a short-lived bearer token."""
    if not _DEEPLYNX_URL or not _DEEPLYNX_API_KEY:
        return None
    result = _make_request(
        "/oauth/token",
        method="POST",
        body={"x_api_key": _DEEPLYNX_API_KEY, "x_api_secret": _DEEPLYNX_API_SECRET},
    )
    if result and isinstance(result.get("value"), str):
        return result["value"]
    return None


def list_projects(container_id: str | None = None) -> list[dict[str, Any]]:
    """Return all projects (containers) the service account can access.

    Args:
        container_id: Optional container ID to filter to a single project.

    Returns:
        List of project dicts with at minimum ``id`` and ``name`` keys.
    """
    if not _DEEPLYNX_URL:
        return [{"id": "stub-project", "name": "MSR Demo Project (stub)"}]

    path = "/containers" if not container_id else f"/containers/{container_id}"
    token = _get_bearer_token()
    result = _make_request(path, token=token)
    if result and isinstance(result.get("value"), list):
        return result["value"]
    return []


def list_datasources(container_id: str) -> list[dict[str, Any]]:
    """Return data sources registered under *container_id*.

    Args:
        container_id: DeepLynx container (project) ID.

    Returns:
        List of data source dicts.
    """
    if not _DEEPLYNX_URL:
        return [{"id": "stub-ds", "name": "Stub Data Source", "type": "standard"}]

    token = _get_bearer_token()
    result = _make_request(
        f"/containers/{container_id}/import/datasources", token=token
    )
    if result and isinstance(result.get("value"), list):
        return result["value"]
    return []


def get_records(
    container_id: str,
    datasource_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Fetch data records from DeepLynx.

    Args:
        container_id: DeepLynx container ID.
        datasource_id: Optional data source filter.
        limit: Max records to return.
        offset: Pagination offset.

    Returns:
        List of record dicts, each with ``id``, ``type``, and ``properties``.
    """
    if not _DEEPLYNX_URL:
        return _STUB_RECORDS[:limit]

    token = _get_bearer_token()
    path = f"/containers/{container_id}/data"
    params = {"limit": limit, "offset": offset}
    if datasource_id:
        params["datasource_id"] = datasource_id
    qs = urllib.parse.urlencode(params)
    result = _make_request(f"{path}?{qs}", token=token)
    if result and isinstance(result.get("value"), list):
        return result["value"]
    return []


def get_timeseries(
    container_id: str,
    datasource_id: str,
    start_time: str | None = None,
    end_time: str | None = None,
    last_n: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch timeseries readings from a DeepLynx timeseries data source.

    Args:
        container_id: DeepLynx container ID.
        datasource_id: Timeseries data source ID.
        start_time: ISO 8601 start timestamp (optional).
        end_time: ISO 8601 end timestamp (optional).
        last_n: Return last N rows (overrides time range when set).

    Returns:
        List of reading dicts with ``timestamp``, ``sensor``, and ``value``.
    """
    if not _DEEPLYNX_URL:
        rows = _STUB_TIMESERIES
        if last_n:
            rows = rows[-last_n:]
        return rows

    token = _get_bearer_token()
    path = f"/containers/{container_id}/import/datasources/{datasource_id}/data"
    params: dict[str, Any] = {}
    if start_time:
        params["start_time"] = start_time
    if end_time:
        params["end_time"] = end_time
    if last_n:
        params["limit"] = last_n
    qs = urllib.parse.urlencode(params) if params else ""
    result = _make_request(f"{path}?{qs}" if qs else path, token=token)
    if result and isinstance(result.get("value"), list):
        return result["value"]
    return []


def create_record(
    container_id: str,
    datasource_id: str,
    type_name: str,
    properties: dict[str, Any],
) -> dict[str, Any] | None:
    """Create a new typed record in DeepLynx.

    Args:
        container_id: Target container.
        datasource_id: Target data source.
        type_name: Metatype name (e.g. ``"Publication"``).
        properties: Key/value properties for the record.

    Returns:
        Created record dict or ``None`` on failure.
    """
    if not _DEEPLYNX_URL:
        stub = {"id": f"stub-new-{type_name}", "type": type_name, "properties": properties}
        logger.info("Stub create_record: %s", stub["id"])
        return stub

    token = _get_bearer_token()
    path = (
        f"/containers/{container_id}/import/datasources/{datasource_id}/imports"
    )
    payload = [{"metatype_name": type_name, "properties": properties}]
    result = _make_request(path, method="POST", body=payload, token=token)
    if result and isinstance(result.get("value"), list) and result["value"]:
        return result["value"][0]
    return None


def get_schema_description(
    container_id: str, datasource_id: str
) -> str:
    """Return a human-readable description of the timeseries schema.

    Used to help the LLM generate valid SQL-like queries against the
    DeepLynx timeseries endpoint.

    Args:
        container_id: Container ID.
        datasource_id: Timeseries data source ID.

    Returns:
        Plain-text schema description string.
    """
    if not _DEEPLYNX_URL:
        return (
            "Table: timeseries_data\n"
            "Columns: timestamp (ISO8601), sensor (TEXT), value (REAL), unit (TEXT)\n"
            "Example sensors: core_temp_c, reactor_power_mw, salt_flow_kg_s\n"
        )

    token = _get_bearer_token()
    path = (
        f"/containers/{container_id}/import/datasources/{datasource_id}"
    )
    result = _make_request(path, token=token)
    if result and isinstance(result.get("value"), dict):
        cfg = result["value"].get("config", {})
        columns = cfg.get("columns", [])
        col_desc = ", ".join(
            f"{c.get('column_name', '?')} ({c.get('type', '?')})"
            for c in columns
        )
        return f"Table: timeseries_data\nColumns: {col_desc or 'timestamp, value'}\n"
    return "Table: timeseries_data\nColumns: timestamp (ISO8601), value (REAL)\n"
