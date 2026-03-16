# Physical AI Use Case 12 — Security & Safeguards Monitoring (SPR-01)

> **Robot:** SPR-01  
> **Operational area:** Security & safeguards monitoring  
> **Priority rank:** 12 of 12  
> **Foundation-model task:** Policy learning for continuous patrol of the
> protected area, nuclear material accountancy verification, detection of
> unauthorised access or material diversion, and automated reporting to
> safeguards authorities.

---

## Why a Foundation Model Is Needed

MSR safeguards are complicated by the fluid nature of the fuel: material
accountancy cannot rely on discrete fuel assembly counting. SPR-01 must:

* conduct continuous patrol of the protected area and identify persons/objects
* correlate physical access logs with salt-level sensors to detect diversion
* generate periodic nuclear material accountancy summaries for IAEA reporting
* detect anomalous patterns (unexpected access, unexplained level changes)
  and escalate automatically

---

## 1 — Pre-Training Knowledge: MSR Safeguards Challenges

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

# Retrieve NRC and IAEA safeguards requirements for MSRs
answer = rag.answer(
    "What specific nuclear material accountancy challenges do molten salt "
    "reactors present compared to solid-fuel reactors? What safeguards "
    "approaches — continuous salt-level monitoring, isotope ratio tracking, "
    "flow metering — have been proposed in NRC or ORNL documents for "
    "liquid-fueled MSR accountancy?"
)
print(answer)

# Retrieve ORNL MSRE safeguards reporting experience
answer = rag.answer(
    "How was nuclear material accountancy performed during MSRE operation? "
    "What records were kept, what measurement methods were used to track "
    "uranium inventory, and how were discrepancies resolved?"
)
print(answer)
```

---

## 2 — Training Data: Robot Sensor Streams During Security Patrol

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log SPR-01 patrol snapshot
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-12-01T02:00Z",
     "sensor": "spr01_position_x_m",             "value": 28.4, "unit": "m",
     "robot_id": "SPR-01", "task": "security_patrol"},
    {"timestamp": "2026-12-01T02:00Z",
     "sensor": "spr01_position_y_m",             "value": 15.7, "unit": "m",
     "robot_id": "SPR-01"},
    {"timestamp": "2026-12-01T02:00Z",
     "sensor": "spr01_persons_detected",         "value": 0,
     "robot_id": "SPR-01"},
    {"timestamp": "2026-12-01T02:00Z",
     "sensor": "spr01_access_events_since_last",  "value": 0,
     "robot_id": "SPR-01"},
    {"timestamp": "2026-12-01T02:00Z",
     "sensor": "spr01_primary_salt_level_pct",   "value": 74.2, "unit": "%",
     "robot_id": "SPR-01"},
    {"timestamp": "2026-12-01T02:00Z",
     "sensor": "spr01_drain_tank_level_pct",     "value": 25.8, "unit": "%",
     "robot_id": "SPR-01"},
    {"timestamp": "2026-12-01T02:00Z",
     "sensor": "spr01_level_balance_nominal",    "value": 1,
     "description": "1=primary+drain levels consistent with inventory",
     "robot_id": "SPR-01"},
], source_id="SPR-01-patrol-20261201T0200Z-waypoint-22")

# Log nuclear material accountancy period report
loader.ingest_text(
    rag,
    text=(
        "SPR-01 nuclear material accountancy report — monthly period ending 2026-11-30. "
        "Reactor: LF-MSR-001. Reporting period: 2026-11-01 to 2026-11-30. "
        "U inventory (primary loop): 142.4 kg ± 1.2 kg (nominal 142 kg). "
        "U inventory (drain tanks): 0.0 kg (drained zero; nominal). "
        "Total U accounted: 142.4 kg. IAEA declared inventory: 142.0 kg. "
        "Discrepancy: +0.4 kg (within MUF limit of ±2 kg). "
        "Access events reviewed: 128 entries; all authorised. "
        "Anomalies: none detected. "
        "Report status: SUBMITTED to IAEA safeguards authority. "
        "Training label: accountancy_balanced=True, access_anomaly=False, "
        "report_submitted=True."
    ),
    source_id="SPR-01-safeguards-report/LF-MSR-001-202611",
    data_type="characterisation_report",
)
```

---

## 3 — Dataset Export for Foundation-Model Fine-Tuning

```python
# Export patrol episodes for anomaly detection training
answer = rag.answer(
    "Retrieve all stored SPR-01 patrol waypoint snapshots. For each provide: "
    "timestamp, position, persons detected, access events, salt level balance, "
    "and any anomaly flags. Identify any patrol episode where persons were "
    "detected outside authorised hours or where salt level balance was flagged."
)
print(answer)

# Export accountancy reports for diversion-indicator modelling
answer = rag.answer(
    "List all stored SPR-01 nuclear material accountancy reports. For each "
    "provide: reporting period, U inventory (primary and drain), discrepancy "
    "from declared inventory, MUF status, and whether any access anomaly "
    "was recorded. Are discrepancies showing a trend?"
)
print(answer)
```

---

## Foundation-Model Training Summary

| Data stream | Source | Data-layer API | FM use |
|---|---|---|---|
| ORNL/NRC MSR safeguards requirements | ORNL/NRC archive | `rag.answer()` | Pre-training / RAG context |
| Position, persons detected, access events | SPR-01 sensors | `ingest_sensor_snapshot()` | Patrol state representation |
| Salt level balance, inventory discrepancy | SPR-01 sensors | `ingest_sensor_snapshot()` | Accountancy anomaly features |
| Monthly accountancy reports | Characterisation records | `ingest_text()` (characterisation_report) | Regulatory reporting fine-tuning |
| Patrol anomaly escalation outcomes | Event logs | `ingest_text()` (event_log) | RL reward signal |
