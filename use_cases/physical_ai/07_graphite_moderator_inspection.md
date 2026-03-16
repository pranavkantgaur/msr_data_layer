# Physical AI Use Case 07 — Graphite Moderator Inspection & Replacement (GIR-01)

> **Robot:** GIR-01  
> **Operational area:** Graphite moderator inspection & replacement  
> **Priority rank:** 7 of 12  
> **Foundation-model task:** Policy learning for visual and dimensional
> inspection of graphite moderator blocks, detection of radiation-induced
> swelling, cracking, and salt impregnation, and coordinated block
> extraction and replacement.

---

## Why a Foundation Model Is Needed

Graphite moderator blocks in an MSR accumulate radiation damage (swelling,
dimensional change, porosity increase) and become impregnated with fuel salt
over their 50-year design lifetime. GIR-01 must:

* navigate the fuel-salt-filled core region using submersible or shielded end-effectors
* image each graphite block surface for crack detection and dimensional change
* compare measured dimensions against as-built records to identify blocks
  exceeding replacement thresholds
* execute block extraction and insertion without disturbing neighbouring blocks

---

## 1 — Pre-Training Knowledge: ORNL Graphite Radiation Damage Data

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

# Retrieve ORNL graphite radiation damage measurements
answer = rag.answer(
    "What dimensional changes, porosity increases, and mechanical property "
    "degradation were measured in graphite moderator samples from the MSRE "
    "after irradiation? At what fluence level or operating duration were "
    "blocks scheduled for replacement? Include ORNL report numbers."
)
print(answer)

# Retrieve salt impregnation data
answer = rag.answer(
    "To what depth did fuel salt penetrate into MSRE graphite moderator "
    "blocks during operation? How was impregnation depth measured, and "
    "what was its effect on neutron moderation efficiency and block structural "
    "integrity?"
)
print(answer)
```

---

## 2 — Training Data: Robot Sensor Streams During Graphite Inspection

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log a GIR-01 graphite block inspection episode
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-05-15T07:00Z",
     "sensor": "gir01_block_id",              "value": "GR-B-042",
     "robot_id": "GIR-01", "task": "graphite_inspection"},
    {"timestamp": "2026-05-15T07:00Z",
     "sensor": "gir01_block_height_mm",       "value": 599.3, "unit": "mm",
     "description": "nominal 600 mm; swelling -0.7 mm",
     "robot_id": "GIR-01"},
    {"timestamp": "2026-05-15T07:00Z",
     "sensor": "gir01_block_width_mm",        "value": 101.8, "unit": "mm",
     "description": "nominal 100 mm; +1.8 mm swelling",
     "robot_id": "GIR-01"},
    {"timestamp": "2026-05-15T07:00Z",
     "sensor": "gir01_surface_crack_count",   "value": 2,
     "robot_id": "GIR-01"},
    {"timestamp": "2026-05-15T07:00Z",
     "sensor": "gir01_max_crack_depth_mm",    "value": 1.4, "unit": "mm",
     "robot_id": "GIR-01"},
    {"timestamp": "2026-05-15T07:00Z",
     "sensor": "gir01_radiation_dose_mGy_h",  "value": 180.0,"unit": "mGy/h",
     "robot_id": "GIR-01"},
], source_id="GIR-01-inspection-GR-B-042-20260515T07Z")

# Log inspection outcome
loader.ingest_text(
    rag,
    text=(
        "GIR-01 inspection result — graphite block GR-B-042. "
        "Date: 2026-05-15. "
        "Dimensional change: width +1.8 mm (threshold +3 mm), height -0.7 mm. "
        "Surface cracks: 2 minor cracks, max depth 1.4 mm (threshold 5 mm). "
        "Salt impregnation depth (estimated from visual): < 0.5 mm (acceptable). "
        "Decision: block serviceable; re-inspect in 18 months. "
        "Training label: replacement_required=False, dimensional_within_limits=True, "
        "crack_within_limits=True, re_inspection_interval_months=18."
    ),
    source_id="GIR-01-episode-log/inspection-GR-B-042-20260515",
    data_type="characterisation_report",
)
```

---

## 3 — Dataset Export for Foundation-Model Fine-Tuning

```python
# Export graphite inspection dataset for anomaly detection training
answer = rag.answer(
    "List all stored GIR-01 graphite block inspection results. For each "
    "provide: block ID, dimensional changes (height, width), crack count and "
    "max depth, salt impregnation estimate, and replacement decision. "
    "Which blocks have been flagged for replacement?"
)
print(answer)
```

---

## Foundation-Model Training Summary

| Data stream | Source | Data-layer API | FM use |
|---|---|---|---|
| ORNL graphite damage fluence limits | ORNL archive | `rag.answer()` | Pre-training / physics priors |
| Block dimensions, crack count/depth | GIR-01 sensors | `ingest_sensor_snapshot()` | State representation |
| Replacement decision labels | Characterisation records | `ingest_text()` (characterisation_report) | Supervised classifier training |
| Block extraction/insertion outcomes | Episode logs | `ingest_text()` (event_log) | Manipulation policy RL |
