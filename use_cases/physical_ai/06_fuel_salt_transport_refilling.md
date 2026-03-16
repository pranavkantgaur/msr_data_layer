# Physical AI Use Case 06 — Fuel Salt Transport & Refilling (FSTR-01)

> **Robot:** FSTR-01  
> **Operational area:** Fuel salt transport & refilling  
> **Priority rank:** 6 of 12  
> **Foundation-model task:** Policy learning for safe, precise transfer of
> molten fluoride fuel salt between storage vessels, drain tanks, and the
> reactor core using remotely operated pumps, valves, and level sensors.

---

## Why a Foundation Model Is Needed

Fuel salt transport requires coordinating pump operation, valve sequencing,
and real-time level monitoring to prevent overfill, underfill, or uncontrolled
criticality changes. FSTR-01 must:

* sequence valve opens and pump ramp-ups in the correct order
* monitor salt level in source and destination vessels continuously
* detect and respond to flow anomalies (blockage, leak, unexpected level change)
* maintain salt temperature above freeze point throughout transfer

---

## 1 — Pre-Training Knowledge: ORNL Salt Transfer Procedures

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

# Retrieve ORNL drain and fill procedures for MSRE
answer = rag.answer(
    "What procedures were used for draining and refilling the MSRE primary "
    "circuit with fuel salt? What valve sequences, pump operations, and "
    "level monitoring steps were involved? What temperature limits were "
    "maintained to prevent salt freezing in transfer lines?"
)
print(answer)

# Retrieve criticality safety constraints on salt volume
answer = rag.answer(
    "What criticality safety constraints governed the volume and enrichment "
    "of fuel salt that could be held in MSRE storage vessels or transfer "
    "lines? What level-sensor interlocks were used?"
)
print(answer)
```

---

## 2 — Training Data: Robot Sensor Streams During Salt Transfer

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log a FSTR-01 fuel salt transfer episode
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-06-20T22:00Z",
     "sensor": "fstr01_source_vessel_level_pct",   "value": 78.4, "unit": "%",
     "robot_id": "FSTR-01", "task": "fuel_salt_transfer"},
    {"timestamp": "2026-06-20T22:00Z",
     "sensor": "fstr01_dest_vessel_level_pct",     "value": 12.1, "unit": "%",
     "robot_id": "FSTR-01"},
    {"timestamp": "2026-06-20T22:00Z",
     "sensor": "fstr01_transfer_pump_rpm",         "value": 850,  "unit": "rpm",
     "robot_id": "FSTR-01"},
    {"timestamp": "2026-06-20T22:00Z",
     "sensor": "fstr01_transfer_line_temp_c",      "value": 512.3,"unit": "°C",
     "robot_id": "FSTR-01"},
    {"timestamp": "2026-06-20T22:00Z",
     "sensor": "fstr01_flow_rate_kg_min",          "value": 4.8,  "unit": "kg/min",
     "robot_id": "FSTR-01"},
    {"timestamp": "2026-06-20T22:00Z",
     "sensor": "fstr01_radiation_dose_mGy_h",      "value": 62.1, "unit": "mGy/h",
     "robot_id": "FSTR-01"},
], source_id="FSTR-01-transfer-20260620T22Z")

# Log transfer completion
loader.ingest_text(
    rag,
    text=(
        "FSTR-01 task episode — fuel salt transfer to drain tank DT-02. "
        "Date: 2026-06-20. Duration: 35 min. "
        "Volume transferred: 420 L. "
        "Transfer line temperature range: 505–525 °C (above freeze limit 420 °C). "
        "Source vessel final level: 42.1%. Destination final level: 58.7%. "
        "No flow anomalies detected. All valve sequences completed in order. "
        "Outcome: PASS. "
        "Training label: task_success=True, flow_nominal=True, "
        "temperature_maintained=True, level_targets_met=True."
    ),
    source_id="FSTR-01-episode-log/transfer-DT02-20260620",
    data_type="event_log",
)
```

---

## 3 — Dataset Export for Foundation-Model Fine-Tuning

```python
# Export labelled transfer episodes for policy training
answer = rag.answer(
    "List all stored FSTR-01 fuel salt transfer episodes. For each provide: "
    "source and destination vessels, volume transferred, transfer line "
    "temperature range, flow rate, duration, and training labels "
    "(task_success, flow_nominal, temperature_maintained). "
    "Flag episodes where transfer line temperature fell below 450 °C."
)
print(answer)
```

---

## Foundation-Model Training Summary

| Data stream | Source | Data-layer API | FM use |
|---|---|---|---|
| ORNL drain/fill procedures & criticality limits | ORNL archive | `rag.answer()` | Pre-training / RAG context |
| Vessel levels, flow rate, line temperature | FSTR-01 sensors | `ingest_sensor_snapshot()` | State representation |
| Transfer episode outcome labels | Episode logs | `ingest_text()` (event_log) | RL reward signal |
| Valve sequencing records | Event logs | `ingest_text()` (event_log) | Imitation learning data |
