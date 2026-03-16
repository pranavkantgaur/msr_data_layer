# Physical AI Use Case 01 — Primary Loop Maintenance & Repair (PLMR-01)

> **Robot:** PLMR-01  
> **Operational area:** Primary loop maintenance & repair  
> **Priority rank:** 1 of 12  
> **Foundation-model task:** Policy learning for remote manipulation of primary
> circuit components (pumps, IHX, piping, valves) under high radiation and
> high-temperature fluoride-salt environments.

---

## Why a Foundation Model Is Needed

The primary loop of a liquid-fueled MSR carries fuel salt at up to 700 °C with
a significant dissolved fission-product inventory. Human access for maintenance
is impossible or severely time-limited. A foundation model for PLMR-01 must:

* identify component condition from visual and thermal camera feeds
* plan multi-step manipulation sequences (valve actuation, pipe coupling,
  pump impeller replacement)
* detect anomalies (salt crust, corrosion pit, mechanical play) and halt
* generate work-order text from completed inspection observations

---

## 1 — Pre-Training Knowledge: ORNL Primary Loop Engineering Data

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

# Retrieve ORNL data on primary loop component failure modes
answer = rag.answer(
    "What mechanical and corrosion failures were observed in MSRE primary "
    "circuit components — pumps, heat exchangers, piping, and valves — during "
    "13 172 hours of operation? Include failure descriptions, operating "
    "conditions at the time, and corrective actions taken."
)
print(answer)

# Retrieve material degradation rates relevant to maintenance scheduling
answer = rag.answer(
    "What corrosion depths and mass-transfer rates were measured for Hastelloy N "
    "and INOR-8 in primary-circuit fluoride salt service at 600–700 °C? "
    "At what point did ORNL engineers schedule component replacement?"
)
print(answer)
```

---

## 2 — Training Data: Robot Sensor Streams During Maintenance Tasks

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log a PLMR-01 maintenance episode: primary pump inspection
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-08-10T09:00Z",
     "sensor": "plmr01_arm_joint_torque_Nm",       "value": [12.4, 8.1, 5.6, 3.2, 1.1, 0.8],
     "unit": "Nm", "robot_id": "PLMR-01", "task": "pump_visual_inspection"},
    {"timestamp": "2026-08-10T09:00Z",
     "sensor": "plmr01_thermal_camera_max_temp_c",  "value": 68.3,
     "unit": "°C", "robot_id": "PLMR-01", "task": "pump_visual_inspection"},
    {"timestamp": "2026-08-10T09:00Z",
     "sensor": "plmr01_radiation_dose_mGy_h",       "value": 14.2,
     "unit": "mGy/h", "robot_id": "PLMR-01"},
    {"timestamp": "2026-08-10T09:00Z",
     "sensor": "plmr01_gripper_force_N",            "value": 45.7,
     "unit": "N", "robot_id": "PLMR-01", "task": "pump_visual_inspection"},
    {"timestamp": "2026-08-10T09:00Z",
     "sensor": "plmr01_battery_pct",                "value": 82.0,
     "unit": "%", "robot_id": "PLMR-01"},
], source_id="PLMR-01-pump-inspection-20260810T09Z")

# Log task outcome as a labelled training episode
loader.ingest_text(
    rag,
    text=(
        "PLMR-01 task episode — primary pump visual inspection. "
        "Date: 2026-08-10. Duration: 42 min. "
        "Observations: minor salt crust on pump casing (2 mm thick, localised). "
        "No mechanical play detected in pump shaft. Thermal profile uniform. "
        "Action taken: crust photographed and measured; maintenance deferred 90 days. "
        "Outcome: PASS. "
        "Training label: task_success=True, anomaly_detected=True (minor crust), "
        "intervention_required=False."
    ),
    source_id="PLMR-01-episode-log/pump-inspection-20260810",
    data_type="event_log",
)
```

---

## 3 — Dataset Export for Foundation-Model Fine-Tuning

```python
# Retrieve all labelled PLMR-01 episodes for offline RL training
answer = rag.answer(
    "List all stored PLMR-01 task episodes. For each episode provide: "
    "task type, duration, anomalies detected, action taken, outcome label "
    "(task_success, anomaly_detected, intervention_required), and the "
    "peak radiation dose recorded. Format as a structured list."
)
print(answer)

# Retrieve ORNL failure-mode taxonomy for grounding the model's anomaly classifier
answer = rag.answer(
    "Provide a taxonomy of primary-loop component failure modes documented "
    "in ORNL MSR reports: category, visual description, typical thermal "
    "signature, and recommended inspection interval."
)
print(answer)
```

---

## Foundation-Model Training Summary

| Data stream | Source | Data-layer API | FM use |
|---|---|---|---|
| ORNL failure-mode taxonomy | ORNL archive | `rag.answer()` | Pre-training / RAG context |
| Joint torques, gripper force, thermal camera | PLMR-01 sensors | `ingest_sensor_snapshot()` | State representation |
| Task outcome labels | Episode logs | `ingest_text()` (event_log) | RL reward signal |
| Maintenance procedure text | Work orders | `ingest_text()` (characterisation_report) | Instruction fine-tuning |
