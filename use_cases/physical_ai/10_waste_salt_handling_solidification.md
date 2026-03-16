# Physical AI Use Case 10 — Waste Salt Handling & Solidification (WSHR-01)

> **Robot:** WSHR-01  
> **Operational area:** Waste salt handling & solidification  
> **Priority rank:** 10 of 12  
> **Foundation-model task:** Policy learning for transferring spent or
> contaminated fluoride salt to solidification vessels, monitoring
> solidification progress, and preparing solid waste forms for interim storage.

---

## Why a Foundation Model Is Needed

Spent fluoride salt containing dissolved fission products and activation
products must be safely immobilised before long-term storage. WSHR-01 must:

* transfer molten waste salt to solidification vessels while maintaining
  temperature above freeze point
* monitor cooling rate to ensure controlled solidification without cracking
* verify final solid form integrity (surface, density, containment)
* generate waste characterisation records for regulatory submission

---

## 1 — Pre-Training Knowledge: ORNL Waste Salt Disposition Data

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

# Retrieve ORNL waste salt characterisation and disposal approaches
answer = rag.answer(
    "What was the composition, activity level, and proposed disposal pathway "
    "for the spent fuel salt and flush salt removed from the MSRE? How was "
    "it characterised before storage, and what solidification or encapsulation "
    "methods were considered? Include ORNL report numbers."
)
print(answer)

# Retrieve fluoride salt solidification thermal data
answer = rag.answer(
    "What are the solidification temperatures, density changes, and "
    "thermal conductivities of FLiBe and FLiNaK fluoride salts during "
    "controlled cooling? What cooling rates avoid thermal cracking of "
    "large salt blocks?"
)
print(answer)
```

---

## 2 — Training Data: Robot Sensor Streams During Waste Solidification

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log a WSHR-01 waste salt solidification episode
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-11-20T10:00Z",
     "sensor": "wshr01_vessel_id",             "value": "WV-007",
     "robot_id": "WSHR-01", "task": "waste_solidification"},
    {"timestamp": "2026-11-20T10:00Z",
     "sensor": "wshr01_salt_temp_c",           "value": 498.2, "unit": "°C",
     "robot_id": "WSHR-01"},
    {"timestamp": "2026-11-20T10:00Z",
     "sensor": "wshr01_cooling_rate_degC_h",   "value": -8.4,  "unit": "°C/h",
     "description": "target: -5 to -10 °C/h",
     "robot_id": "WSHR-01"},
    {"timestamp": "2026-11-20T10:00Z",
     "sensor": "wshr01_solidification_front_mm","value": 42.0, "unit": "mm",
     "description": "depth of solidified layer from vessel wall",
     "robot_id": "WSHR-01"},
    {"timestamp": "2026-11-20T10:00Z",
     "sensor": "wshr01_surface_crack_detected", "value": 0,
     "robot_id": "WSHR-01"},
    {"timestamp": "2026-11-20T10:00Z",
     "sensor": "wshr01_radiation_dose_mGy_h",  "value": 95.3, "unit": "mGy/h",
     "robot_id": "WSHR-01"},
], source_id="WSHR-01-solidification-WV007-20261120T10Z")

# Log solidification completion and waste form characterisation
loader.ingest_text(
    rag,
    text=(
        "WSHR-01 task episode — waste salt solidification, vessel WV-007. "
        "Date: 2026-11-20 to 2026-11-21. Total cooling time: 18 h. "
        "Final salt temperature: 28 °C (fully solidified). "
        "Average cooling rate: -8.1 °C/h (within target -5 to -10 °C/h). "
        "Surface integrity: no cracks detected on visual inspection. "
        "Estimated activity: 2.4 TBq total (dominated by ¹³⁷Cs, ⁹⁰Sr). "
        "Waste form dimensions: 320 mm diameter × 450 mm height. "
        "Outcome: PASS — waste form accepted for interim storage. "
        "Training label: task_success=True, crack_free=True, "
        "cooling_rate_controlled=True, form_accepted=True."
    ),
    source_id="WSHR-01-episode-log/solidification-WV007-20261120",
    data_type="characterisation_report",
)
```

---

## 3 — Dataset Export for Foundation-Model Fine-Tuning

```python
# Export solidification episodes for process control policy training
answer = rag.answer(
    "List all stored WSHR-01 waste salt solidification episodes. For each "
    "provide: vessel ID, cooling rate achieved, total cooling time, "
    "crack detection result, and waste form acceptance decision. "
    "Flag any episodes where cooling rate exceeded -10 °C/h or cracks "
    "were detected."
)
print(answer)
```

---

## Foundation-Model Training Summary

| Data stream | Source | Data-layer API | FM use |
|---|---|---|---|
| ORNL waste salt composition & disposal data | ORNL archive | `rag.answer()` | Pre-training / RAG context |
| Salt temperature, cooling rate, solidification front | WSHR-01 sensors | `ingest_sensor_snapshot()` | State representation |
| Waste form acceptance labels | Characterisation records | `ingest_text()` (characterisation_report) | Supervised classifier |
| Cooling rate control outcomes | Episode logs | `ingest_text()` (event_log) | Process control RL reward |
