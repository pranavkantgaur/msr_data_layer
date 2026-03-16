# Physical AI Use Case 04 — Radiation Mapping & Autonomous Inspection (RMR-01)

> **Robot:** RMR-01  
> **Operational area:** Radiation mapping & autonomous inspection  
> **Priority rank:** 4 of 12  
> **Foundation-model task:** Policy learning for autonomous navigation, dose-rate
> mapping, and visual anomaly detection across the full reactor building including
> high-dose zones inaccessible to human workers.

---

## Why a Foundation Model Is Needed

Radiation fields in an MSR building are spatially heterogeneous and
time-varying (following fuel redistribution, component changes, and decay).
RMR-01 must:

* navigate complex 3-D environments autonomously (ramps, doors, confined spaces)
* build real-time dose-rate maps and fuse them with building floor plans
* detect visual anomalies (salt leaks, pipe deformation, unexpected deposits)
* plan inspection routes that minimise cumulative dose while maximising coverage

---

## 1 — Pre-Training Knowledge: ORNL Radiation Field Data

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

# Retrieve ORNL radiation field measurements from MSRE
answer = rag.answer(
    "What dose rates were measured at various locations in and around the "
    "MSRE building during operation and after shutdown? Include locations "
    "(primary cell, drain tank cell, off-gas cell, access corridors), "
    "dose rates in mGy/h or R/h, and how they changed with reactor state."
)
print(answer)

# Retrieve ORNL guidance on hotspot identification
answer = rag.answer(
    "What locations in the MSRE consistently showed elevated radiation fields "
    "due to noble-metal fission product deposition, salt leaks, or activated "
    "structural materials? How were these hotspots identified and managed?"
)
print(answer)
```

---

## 2 — Training Data: Robot Sensor Streams During Inspection

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log a RMR-01 radiation mapping run
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-07-20T08:00Z",
     "sensor": "rmr01_gps_x_m",            "value": 12.4,  "unit": "m",
     "robot_id": "RMR-01", "task": "radiation_mapping"},
    {"timestamp": "2026-07-20T08:00Z",
     "sensor": "rmr01_gps_y_m",            "value": 8.7,   "unit": "m",
     "robot_id": "RMR-01"},
    {"timestamp": "2026-07-20T08:00Z",
     "sensor": "rmr01_gps_z_m",            "value": 2.1,   "unit": "m",
     "robot_id": "RMR-01"},
    {"timestamp": "2026-07-20T08:00Z",
     "sensor": "rmr01_dose_rate_mGy_h",    "value": 42.8,  "unit": "mGy/h",
     "robot_id": "RMR-01"},
    {"timestamp": "2026-07-20T08:00Z",
     "sensor": "rmr01_cumulative_dose_mGy","value": 3.14,  "unit": "mGy",
     "robot_id": "RMR-01"},
    {"timestamp": "2026-07-20T08:00Z",
     "sensor": "rmr01_visual_anomaly_flag", "value": 0,
     "robot_id": "RMR-01"},
    {"timestamp": "2026-07-20T08:00Z",
     "sensor": "rmr01_battery_pct",         "value": 74.0, "unit": "%",
     "robot_id": "RMR-01"},
], source_id="RMR-01-mapping-20260720T08Z-waypoint-14")

# Log a hotspot detection event
loader.ingest_text(
    rag,
    text=(
        "RMR-01 inspection event — hotspot detected. "
        "Date: 2026-07-20T09:14Z. Location: primary cell, NE corner, (14.2, 9.1, 1.8) m. "
        "Peak dose rate: 380 mGy/h (background in zone: 45 mGy/h). "
        "Visual observation: small orange-white salt deposit on pipe flange, "
        "approximately 3 cm diameter. "
        "Action: location flagged on dose map; PLMR-01 maintenance ticket raised. "
        "Training label: anomaly_type=salt_leak, severity=minor, "
        "escalation_required=True."
    ),
    source_id="RMR-01-event-log/hotspot-20260720T0914Z",
    data_type="event_log",
)
```

---

## 3 — Dataset Export for Foundation-Model Fine-Tuning

```python
# Export dose-map dataset for navigation policy training
answer = rag.answer(
    "Retrieve all stored RMR-01 waypoint sensor records from 2026-07-20. "
    "For each waypoint provide: (x, y, z) coordinates, dose rate, cumulative "
    "dose, and visual anomaly flag. Sort by timestamp. "
    "This forms a dose-map training episode."
)
print(answer)

# Export anomaly detection labels
answer = rag.answer(
    "List all RMR-01 anomaly detection events stored in the data layer. "
    "For each provide: timestamp, location, peak dose rate, anomaly type, "
    "severity, and whether escalation was triggered. "
    "These labels train the visual anomaly classifier."
)
print(answer)
```

---

## Foundation-Model Training Summary

| Data stream | Source | Data-layer API | FM use |
|---|---|---|---|
| ORNL radiation field maps & hotspot data | ORNL archive | `rag.answer()` | Pre-training / RAG context |
| (x, y, z), dose rate, cumulative dose | RMR-01 sensors | `ingest_sensor_snapshot()` | Navigation state representation |
| Hotspot location, anomaly type, severity | Event logs | `ingest_text()` (event_log) | Anomaly classifier training |
| Inspection route outcomes | Episode logs | `ingest_text()` (event_log) | Route-planning RL reward |
