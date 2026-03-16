# Physical AI Use Case 02 — Hot-Cell Chemical Processing Automation (HCPR-01)

> **Robot:** HCPR-01  
> **Operational area:** Hot-cell chemical processing automation  
> **Priority rank:** 2 of 12  
> **Foundation-model task:** Policy learning for precise liquid handling,
> chemical reagent dosing, and analytical instrument operation inside shielded
> hot cells with fluoride salt samples and fission-product concentrates.

---

## Why a Foundation Model Is Needed

Hot-cell chemical processing involves manipulating highly radioactive fluoride
salt samples for redox adjustment, fission-product separation, and chemical
analysis. HCPR-01 must:

* pipette and dilute molten or dissolved salt samples with sub-millilitre precision
* operate analytical instruments (ICP-OES, electrochemical cells, titration)
* monitor real-time chemistry (UF₃/UF₄ ratio, redox potential, pH equivalents)
* adapt procedures when readings deviate from expected ranges

---

## 1 — Pre-Training Knowledge: ORNL Hot-Cell Chemistry Procedures

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

# Retrieve ORNL redox chemistry procedures and UF₃/UF₄ control data
answer = rag.answer(
    "What chemical processing procedures were used in the MSRE hot cells to "
    "control the UF₃/UF₄ redox ratio in FLiBe fuel salt? Include reagent "
    "types, dosing sequences, target redox potentials, and ORNL report numbers."
)
print(answer)

# Retrieve fission-product separation chemistry
answer = rag.answer(
    "What chemical separation or precipitation methods were used or proposed "
    "in ORNL MSR reports to remove fission products (noble metals, rare earths, "
    "tellurium) from molten fluoride fuel salt? What temperature and reagent "
    "conditions were required?"
)
print(answer)
```

---

## 2 — Training Data: Robot Sensor Streams During Chemical Processing

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log a HCPR-01 episode: UF₃/UF₄ redox adjustment
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-08-12T14:00Z",
     "sensor": "hcpr01_pipette_volume_dispensed_ul", "value": 250.0,
     "unit": "µL", "robot_id": "HCPR-01", "task": "redox_adjustment"},
    {"timestamp": "2026-08-12T14:00Z",
     "sensor": "hcpr01_cell_redox_potential_mV",     "value": -312.4,
     "unit": "mV", "robot_id": "HCPR-01", "task": "redox_adjustment"},
    {"timestamp": "2026-08-12T14:00Z",
     "sensor": "hcpr01_sample_temp_c",               "value": 48.3,
     "unit": "°C", "robot_id": "HCPR-01"},
    {"timestamp": "2026-08-12T14:00Z",
     "sensor": "hcpr01_radiation_dose_mGy_h",        "value": 220.1,
     "unit": "mGy/h", "robot_id": "HCPR-01"},
    {"timestamp": "2026-08-12T14:00Z",
     "sensor": "hcpr01_reagent_HF_remaining_mL",     "value": 18.6,
     "unit": "mL", "robot_id": "HCPR-01"},
], source_id="HCPR-01-redox-adjustment-20260812T14Z")

# Log episode outcome
loader.ingest_text(
    rag,
    text=(
        "HCPR-01 task episode — UF₃/UF₄ redox adjustment. "
        "Date: 2026-08-12. Duration: 28 min. "
        "Initial redox potential: -285 mV (too oxidising). "
        "Reagent added: 250 µL of UF₃ solution. "
        "Final redox potential: -312 mV (within target range −300 to −330 mV). "
        "ICP-OES check: U concentration 42.3 g/L (nominal). "
        "Outcome: PASS. "
        "Training label: task_success=True, reagent_dose_correct=True, "
        "target_reached=True."
    ),
    source_id="HCPR-01-episode-log/redox-adjustment-20260812",
    data_type="event_log",
)
```

---

## 3 — Dataset Export for Foundation-Model Fine-Tuning

```python
# Export labelled HCPR-01 chemical-processing episodes
answer = rag.answer(
    "List all stored HCPR-01 chemical processing episodes. For each episode "
    "provide: procedure type, initial and final redox potential, reagent "
    "volume dispensed, ICP-OES result if available, and training labels "
    "(task_success, target_reached). Format as a structured list."
)
print(answer)

# Retrieve ORNL redox tolerance windows for model reward shaping
answer = rag.answer(
    "What are the acceptable UF₃/UF₄ ratio ranges and corresponding redox "
    "potential windows specified in ORNL MSR reports for safe fuel salt "
    "operation? Include upper and lower limits and the consequences of "
    "exceeding each limit."
)
print(answer)
```

---

## Foundation-Model Training Summary

| Data stream | Source | Data-layer API | FM use |
|---|---|---|---|
| ORNL redox chemistry procedures | ORNL archive | `rag.answer()` | Pre-training / RAG context |
| Pipette volumes, redox potential, temperature | HCPR-01 sensors | `ingest_sensor_snapshot()` | State representation |
| Episode outcomes, dosing accuracy labels | Episode logs | `ingest_text()` (event_log) | RL reward signal |
| ICP-OES analysis results | Characterisation records | `ingest_text()` (characterisation_report) | Supervised fine-tuning |
