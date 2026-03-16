# Physical AI Use Case 11 — External Structural Inspection (SIR-01)

> **Robot:** SIR-01  
> **Operational area:** External structural inspection  
> **Priority rank:** 11 of 12  
> **Foundation-model task:** Policy learning for autonomous survey of the
> reactor building exterior, roof, containment vessel surfaces, and
> below-grade structures to detect concrete degradation, weld defects,
> corrosion, and settlement anomalies.

---

## Why a Foundation Model Is Needed

Reactor building structural integrity is a long-term safety requirement.
Periodic external inspection by SIR-01 must:

* navigate exterior surfaces including vertical walls, roof edges, and
  below-grade vaults using climbing or wheeled platforms
* detect concrete cracking, spalling, rebar corrosion staining, and
  joint sealant failure
* compare observations against baseline as-built records and prior inspection images
* generate structured inspection reports for regulatory and civil engineering teams

---

## 1 — Pre-Training Knowledge: MSR Structural Requirements

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

# Retrieve MSR containment and structural design requirements
answer = rag.answer(
    "What structural integrity requirements and inspection intervals are "
    "specified or referenced in MSR regulatory and design documents for "
    "reactor containment buildings? What visual or non-destructive testing "
    "criteria were applied to MSRE structures?"
)
print(answer)

# Retrieve concrete degradation mechanisms relevant to MSR environments
answer = rag.answer(
    "What concrete degradation mechanisms — radiation-induced volume change, "
    "thermal cycling, chemical attack from fluoride vapours — are documented "
    "in ORNL MSR or NRC reports for reactor building structures? At what "
    "crack width or spall depth is remediation required?"
)
print(answer)
```

---

## 2 — Training Data: Robot Sensor Streams During Structural Inspection

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log a SIR-01 exterior inspection waypoint
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-04-10T09:30Z",
     "sensor": "sir01_position_x_m",          "value": 45.2,  "unit": "m",
     "robot_id": "SIR-01", "task": "exterior_structural_inspection"},
    {"timestamp": "2026-04-10T09:30Z",
     "sensor": "sir01_position_y_m",          "value": 12.8,  "unit": "m",
     "robot_id": "SIR-01"},
    {"timestamp": "2026-04-10T09:30Z",
     "sensor": "sir01_surface_height_m",      "value": 8.4,   "unit": "m",
     "robot_id": "SIR-01"},
    {"timestamp": "2026-04-10T09:30Z",
     "sensor": "sir01_camera_resolution_mm_px","value": 0.3,  "unit": "mm/px",
     "robot_id": "SIR-01"},
    {"timestamp": "2026-04-10T09:30Z",
     "sensor": "sir01_crack_detected",        "value": 1,
     "robot_id": "SIR-01"},
    {"timestamp": "2026-04-10T09:30Z",
     "sensor": "sir01_crack_width_mm",        "value": 0.4,   "unit": "mm",
     "robot_id": "SIR-01"},
    {"timestamp": "2026-04-10T09:30Z",
     "sensor": "sir01_crack_length_mm",       "value": 85.0,  "unit": "mm",
     "robot_id": "SIR-01"},
], source_id="SIR-01-inspection-20260410T0930Z-waypoint-38")

# Log inspection finding
loader.ingest_text(
    rag,
    text=(
        "SIR-01 structural finding — south wall, elevation 8.4 m. "
        "Date: 2026-04-10. "
        "Finding: hairline crack, width 0.4 mm, length 85 mm, "
        "orientation: vertical, no rebar staining adjacent. "
        "Classification: Category B (monitor; below repair threshold of 0.5 mm). "
        "Comparison with 2024 baseline: crack not present in 2024 inspection "
        "(new crack; growth monitoring initiated). "
        "Next inspection: 6 months. "
        "Training label: defect_type=hairline_crack, severity=minor, "
        "remediation_required=False, monitoring_required=True."
    ),
    source_id="SIR-01-finding-log/south-wall-8.4m-20260410",
    data_type="characterisation_report",
)
```

---

## 3 — Dataset Export for Foundation-Model Fine-Tuning

```python
# Export structural inspection findings for defect classifier training
answer = rag.answer(
    "List all stored SIR-01 structural inspection findings. For each provide: "
    "location (x, y, height), defect type, crack width and length if applicable, "
    "severity classification, and training labels (defect_type, severity, "
    "remediation_required). Summarise the distribution of defect types found."
)
print(answer)
```

---

## Foundation-Model Training Summary

| Data stream | Source | Data-layer API | FM use |
|---|---|---|---|
| MSR structural requirements & defect thresholds | ORNL/NRC archive | `rag.answer()` | Pre-training / RAG context |
| Position, camera resolution, crack geometry | SIR-01 sensors | `ingest_sensor_snapshot()` | State representation |
| Defect classification, severity labels | Characterisation records | `ingest_text()` (characterisation_report) | Visual defect classifier |
| Inspection route outcomes | Episode logs | `ingest_text()` (event_log) | Route-planning RL reward |
