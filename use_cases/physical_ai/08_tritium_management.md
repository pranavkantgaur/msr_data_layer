# Physical AI Use Case 08 — Tritium Management Systems (TMR-01)

> **Robot:** TMR-01  
> **Operational area:** Tritium management systems  
> **Priority rank:** 8 of 12  
> **Foundation-model task:** Policy learning for monitoring tritium permeation
> barriers, operating tritium extraction and immobilisation equipment, and
> detecting tritium breaches before they reach regulatory release limits.

---

## Why a Foundation Model Is Needed

Tritium is produced in MSRs primarily from ⁶Li(n,α)T reactions in the FLiBe
salt. It permeates through metallic walls into coolant and containment
systems. TMR-01 must:

* monitor tritium concentration sensors at multiple circuit boundaries
* operate getter beds, isotope separation columns, and immobilisation systems
* detect permeation rate anomalies that indicate barrier degradation
* generate regulatory dose-assessment reports from accumulated data

---

## 1 — Pre-Training Knowledge: ORNL Tritium Data from MSRE

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

# Retrieve ORNL tritium production and permeation measurements
answer = rag.answer(
    "What tritium production rates, permeation rates through Hastelloy N, "
    "and measured tritium concentrations in the MSRE coolant air and "
    "steam systems were reported in ORNL documents? Include measurement "
    "methods and regulatory release limits applied."
)
print(answer)

# Retrieve ORNL tritium management strategies
answer = rag.answer(
    "What methods were proposed or tested in ORNL MSR reports for capturing "
    "tritium from the coolant salt or off-gas stream? Include getter materials, "
    "operating temperatures, capture efficiencies, and saturation limits."
)
print(answer)
```

---

## 2 — Training Data: Robot Sensor Streams During Tritium Management

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log TMR-01 routine tritium monitoring snapshot
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-09-10T00:00Z",
     "sensor": "tmr01_primary_tritium_Bq_mL",      "value": 1.42e6,
     "unit": "Bq/mL", "robot_id": "TMR-01", "task": "tritium_monitoring"},
    {"timestamp": "2026-09-10T00:00Z",
     "sensor": "tmr01_secondary_tritium_Bq_mL",    "value": 3.8e4,
     "unit": "Bq/mL", "robot_id": "TMR-01"},
    {"timestamp": "2026-09-10T00:00Z",
     "sensor": "tmr01_off_gas_tritium_Bq_m3",      "value": 2.1e5,
     "unit": "Bq/m³", "robot_id": "TMR-01"},
    {"timestamp": "2026-09-10T00:00Z",
     "sensor": "tmr01_getter_bed_saturation_pct",  "value": 34.2,
     "unit": "%", "robot_id": "TMR-01"},
    {"timestamp": "2026-09-10T00:00Z",
     "sensor": "tmr01_barrier_permeation_Bq_h",    "value": 1.8e4,
     "unit": "Bq/h", "robot_id": "TMR-01"},
    {"timestamp": "2026-09-10T00:00Z",
     "sensor": "tmr01_regulatory_limit_pct",       "value": 12.4,
     "unit": "%",
     "description": "secondary tritium as % of regulatory limit",
     "robot_id": "TMR-01"},
], source_id="TMR-01-monitoring-20260910T00Z")

# Log getter bed replacement event
loader.ingest_text(
    rag,
    text=(
        "TMR-01 event — getter bed GB-03 saturation threshold reached. "
        "Timestamp: 2026-10-01T14:30Z. "
        "Getter bed saturation: 78% (threshold: 75%). "
        "Action: getter bed GB-03 isolated; GB-04 brought online. "
        "Switchover time: 4.2 min. Off-gas tritium transient: +8% for 3 min. "
        "No regulatory limit approached. "
        "Training label: task_success=True, switchover_time_acceptable=True, "
        "regulatory_limit_maintained=True."
    ),
    source_id="TMR-01-event-log/getter-replacement-20261001T1430Z",
    data_type="event_log",
)
```

---

## 3 — Dataset Export for Foundation-Model Fine-Tuning

```python
# Export tritium monitoring time-series for anomaly detection
answer = rag.answer(
    "Retrieve all stored TMR-01 tritium monitoring snapshots. For each "
    "provide: timestamp, primary tritium concentration, secondary tritium "
    "concentration, off-gas tritium, getter bed saturation, and regulatory "
    "limit utilisation percentage. Flag any readings where secondary tritium "
    "exceeded 30% of regulatory limit."
)
print(answer)
```

---

## Foundation-Model Training Summary

| Data stream | Source | Data-layer API | FM use |
|---|---|---|---|
| ORNL tritium production & permeation data | ORNL archive | `rag.answer()` | Pre-training / physics priors |
| Primary/secondary/off-gas tritium concentrations | TMR-01 sensors | `ingest_sensor_snapshot()` | State representation |
| Getter bed saturation, permeation rate | TMR-01 sensors | `ingest_sensor_snapshot()` | Anomaly feature inputs |
| Getter bed replacement outcomes | Event logs | `ingest_text()` (event_log) | RL reward signal |
