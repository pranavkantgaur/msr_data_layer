# Physical AI Use Case 09 — Off-Gas System Handling (OGSR-01)

> **Robot:** OGSR-01  
> **Operational area:** Off-gas system handling  
> **Priority rank:** 9 of 12  
> **Foundation-model task:** Policy learning for monitoring and controlling
> the off-gas processing train that captures volatile fission products
> (Kr, Xe, iodine, tritium) stripped from the fuel salt, and for performing
> maintenance on charcoal delay beds and filtration stages.

---

## Why a Foundation Model Is Needed

The MSR off-gas system removes radioactive gases from the fuel salt before
they can build up and create a safety or regulatory hazard. OGSR-01 must:

* monitor gas flows, pressures, and radioactivity at multiple points in the
  off-gas train
* detect charcoal bed breakthrough, filter clogging, or unexpected activity spikes
* coordinate valve switching to bring spare beds or filters online
* perform visual inspection and mechanical maintenance of compressors and
  filter housings under elevated radiation

---

## 1 — Pre-Training Knowledge: ORNL Off-Gas System Data

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

# Retrieve ORNL off-gas system design and operational data
answer = rag.answer(
    "How was the MSRE off-gas system designed to capture noble gas fission "
    "products stripped from the fuel salt? What charcoal delay bed dimensions, "
    "hold-up times, and operating temperatures were used? What activity levels "
    "were measured at the stack?"
)
print(answer)

# Retrieve off-gas system failure modes
answer = rag.answer(
    "What failures or anomalies were recorded in the MSRE off-gas system "
    "during operation? Include charcoal bed breakthrough events, compressor "
    "failures, and any unexpected radioactivity releases to the building."
)
print(answer)
```

---

## 2 — Training Data: Robot Sensor Streams During Off-Gas Operations

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log a OGSR-01 off-gas monitoring snapshot
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-07-05T18:00Z",
     "sensor": "ogsr01_gas_flow_inlet_slpm",       "value": 2.8,   "unit": "slpm",
     "robot_id": "OGSR-01", "task": "off_gas_monitoring"},
    {"timestamp": "2026-07-05T18:00Z",
     "sensor": "ogsr01_charcoal_bed_inlet_Bq_m3",  "value": 4.2e9, "unit": "Bq/m³",
     "robot_id": "OGSR-01"},
    {"timestamp": "2026-07-05T18:00Z",
     "sensor": "ogsr01_charcoal_bed_outlet_Bq_m3", "value": 1.8e4, "unit": "Bq/m³",
     "description": "decontamination factor ~2.3e5",
     "robot_id": "OGSR-01"},
    {"timestamp": "2026-07-05T18:00Z",
     "sensor": "ogsr01_iodine_filter_dp_Pa",       "value": 420.0, "unit": "Pa",
     "robot_id": "OGSR-01"},
    {"timestamp": "2026-07-05T18:00Z",
     "sensor": "ogsr01_stack_activity_Bq_m3",      "value": 8.2e2, "unit": "Bq/m³",
     "robot_id": "OGSR-01"},
    {"timestamp": "2026-07-05T18:00Z",
     "sensor": "ogsr01_radiation_dose_mGy_h",      "value": 28.4,  "unit": "mGy/h",
     "robot_id": "OGSR-01"},
], source_id="OGSR-01-monitoring-20260705T18Z")

# Log a charcoal bed switchover event
loader.ingest_text(
    rag,
    text=(
        "OGSR-01 event — charcoal bed CB-1 switchover to CB-2. "
        "Timestamp: 2026-08-14T11:05Z. "
        "Trigger: CB-1 decontamination factor dropped from 2.3e5 to 8.4e4 "
        "(breakthrough threshold: 1e5). "
        "Action: CB-2 brought online; CB-1 isolated for regeneration. "
        "Switchover duration: 6.8 min. Stack activity during switchover: "
        "peak 2.1e3 Bq/m³ (regulatory limit: 1e6 Bq/m³). "
        "Training label: task_success=True, breakthrough_detected=True, "
        "switchover_smooth=True, regulatory_limit_maintained=True."
    ),
    source_id="OGSR-01-event-log/CB1-switchover-20260814T1105Z",
    data_type="event_log",
)
```

---

## 3 — Dataset Export for Foundation-Model Fine-Tuning

```python
# Export off-gas monitoring data for anomaly detection training
answer = rag.answer(
    "Retrieve all stored OGSR-01 off-gas monitoring snapshots. For each "
    "provide: timestamp, inlet and outlet activity, charcoal bed decontamination "
    "factor, iodine filter differential pressure, and stack activity. "
    "Flag any readings where decontamination factor fell below 1e5 or stack "
    "activity exceeded 1e5 Bq/m³."
)
print(answer)
```

---

## Foundation-Model Training Summary

| Data stream | Source | Data-layer API | FM use |
|---|---|---|---|
| ORNL off-gas system design & failure modes | ORNL archive | `rag.answer()` | Pre-training / RAG context |
| Flow, activity, filter dp, stack activity | OGSR-01 sensors | `ingest_sensor_snapshot()` | State representation |
| Charcoal bed breakthrough events | Event logs | `ingest_text()` (event_log) | Anomaly classifier training |
| Bed switchover outcomes | Episode logs | `ingest_text()` (event_log) | Control policy RL reward |
