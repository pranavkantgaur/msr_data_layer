# Physical AI Use Case 05 — Freeze Plug Safety Monitoring (FPMR-01)

> **Robot:** FPMR-01  
> **Operational area:** Freeze plug safety monitoring  
> **Priority rank:** 5 of 12  
> **Foundation-model task:** Policy learning for continuous thermal monitoring
> of freeze plugs, early detection of unintended freeze or thaw events, and
> coordination of corrective heater/cooler adjustments.

---

## Why a Foundation Model Is Needed

The freeze plug is a critical passive safety device: a section of salt
deliberately solidified by active cooling that melts on loss of power to drain
the core. FPMR-01 must:

* monitor freeze plug temperature profiles continuously at sub-minute intervals
* detect early signs of unintended partial thaw (rising temperature gradient)
  or over-freeze (temperature below design minimum)
* recommend or trigger heater/cooler adjustments before the plug transitions
  to an unsafe state
* log all thermal events for regulatory audit

---

## 1 — Pre-Training Knowledge: ORNL Freeze Plug Design Data

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

# Retrieve ORNL freeze plug design and operating parameters
answer = rag.answer(
    "What were the design operating temperatures, cooling requirements, and "
    "melting/freezing behaviour of the freeze plug used in the MSRE drain "
    "system? What temperature margins were maintained between the plug "
    "operating state and the thaw threshold?"
)
print(answer)

# Retrieve ORNL freeze plug incident data
answer = rag.answer(
    "Were there any unintended freeze plug thaw or over-freeze events during "
    "MSRE operation? If so, what caused them, how were they detected, "
    "and what corrective actions were taken? Include ORNL report references."
)
print(answer)
```

---

## 2 — Training Data: Robot Sensor Streams During Freeze Plug Monitoring

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log routine freeze plug thermal monitoring snapshot
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-10-05T12:00Z",
     "sensor": "fpmr01_plug_top_temp_c",      "value": 452.3, "unit": "°C",
     "robot_id": "FPMR-01", "task": "freeze_plug_monitoring"},
    {"timestamp": "2026-10-05T12:00Z",
     "sensor": "fpmr01_plug_mid_temp_c",      "value": 438.7, "unit": "°C",
     "robot_id": "FPMR-01"},
    {"timestamp": "2026-10-05T12:00Z",
     "sensor": "fpmr01_plug_bottom_temp_c",   "value": 421.1, "unit": "°C",
     "robot_id": "FPMR-01"},
    {"timestamp": "2026-10-05T12:00Z",
     "sensor": "fpmr01_cooler_power_kW",      "value": 1.8,   "unit": "kW",
     "robot_id": "FPMR-01"},
    {"timestamp": "2026-10-05T12:00Z",
     "sensor": "fpmr01_plug_state",           "value": "SOLID",
     "robot_id": "FPMR-01"},
    {"timestamp": "2026-10-05T12:00Z",
     "sensor": "fpmr01_thaw_margin_degC",     "value": 27.7,  "unit": "°C",
     "description": "margin below thaw threshold (480 °C)",
     "robot_id": "FPMR-01"},
], source_id="FPMR-01-monitoring-20261005T12Z")

# Log a partial-thaw warning event
loader.ingest_text(
    rag,
    text=(
        "FPMR-01 event — partial thaw warning. "
        "Timestamp: 2026-11-02T03:22Z. "
        "Trigger: plug top temperature rose from 452 °C to 471 °C over 8 min "
        "(rate: 2.4 °C/min, threshold: 1.5 °C/min). "
        "Thaw margin reduced to 9 °C. "
        "Automated response: cooler power increased from 1.8 kW to 3.2 kW. "
        "Temperature stabilised at 463 °C within 6 min. "
        "Cause identified: secondary loop transient (reduced flow). "
        "Outcome: thaw prevented; plug remained SOLID throughout. "
        "Training label: event_type=partial_thaw_warning, "
        "automated_response_effective=True, plug_integrity_maintained=True."
    ),
    source_id="FPMR-01-event-log/partial-thaw-20261102T0322Z",
    data_type="event_log",
)
```

---

## 3 — Dataset Export for Foundation-Model Fine-Tuning

```python
# Export freeze plug thermal time-series for anomaly detection training
answer = rag.answer(
    "Retrieve all FPMR-01 freeze plug monitoring snapshots. For each "
    "provide: timestamp, top/mid/bottom temperatures, cooler power, plug "
    "state, and thaw margin. Flag any readings where thaw margin fell below "
    "15 °C or temperature rate of change exceeded 1.5 °C/min."
)
print(answer)

# Export event labels for policy training
answer = rag.answer(
    "List all FPMR-01 thermal events (warnings, interventions) stored in the "
    "data layer. For each provide: event type, initial and final temperatures, "
    "cooler power adjustment made, time to stabilisation, and training labels."
)
print(answer)
```

---

## Foundation-Model Training Summary

| Data stream | Source | Data-layer API | FM use |
|---|---|---|---|
| ORNL freeze plug design temperatures | ORNL archive | `rag.answer()` | Pre-training / physics priors |
| Top/mid/bottom plug temps, cooler power | FPMR-01 sensors | `ingest_sensor_snapshot()` | State representation |
| Thaw/over-freeze event logs | Event logs | `ingest_text()` (event_log) | Anomaly detection training |
| Cooler adjustment outcomes | Episode logs | `ingest_text()` (event_log) | Control policy RL reward |
