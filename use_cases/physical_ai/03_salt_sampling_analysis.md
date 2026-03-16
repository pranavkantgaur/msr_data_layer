# Physical AI Use Case 03 — Salt Sampling & Analysis (SSR-01)

> **Robot:** SSR-01  
> **Operational area:** Salt sampling & analysis  
> **Priority rank:** 3 of 12  
> **Foundation-model task:** Policy learning for autonomous collection of
> molten fluoride salt micro-samples from the primary and secondary circuits,
> safe transfer to analytical instruments, and interpretation of results.

---

## Why a Foundation Model Is Needed

Periodic salt sampling is essential for monitoring fuel chemistry, fission-product
build-up, and corrosion-product accumulation. SSR-01 must:

* insert a sampling probe into a molten salt stream at 600–700 °C
* collect a reproducible micro-sample volume (1–5 mL) without contamination
* transfer the sample to a cooled containment vessel
* trigger downstream analytical workflows and log results

---

## 1 — Pre-Training Knowledge: ORNL Salt Sampling Procedures

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

# Retrieve ORNL salt sampling methods and frequencies
answer = rag.answer(
    "How were fuel salt samples collected from the MSRE primary circuit during "
    "operation? What sampling frequency, probe design, sample volume, and "
    "handling procedures were used? What parameters were measured in each "
    "sample (UF₃/UF₄, corrosion metals, fission products)?"
)
print(answer)

# Retrieve expected salt chemistry ranges for anomaly detection
answer = rag.answer(
    "What were the normal operating ranges for key salt chemistry parameters "
    "in the MSRE — chromium concentration, uranium oxidation state, noble-metal "
    "fission-product levels — and what deviations triggered corrective action?"
)
print(answer)
```

---

## 2 — Training Data: Robot Sensor Streams During Sampling

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log a SSR-01 sampling episode
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-09-01T06:00Z",
     "sensor": "ssr01_probe_insertion_depth_mm",  "value": 142.0,
     "unit": "mm", "robot_id": "SSR-01", "task": "primary_salt_sampling"},
    {"timestamp": "2026-09-01T06:00Z",
     "sensor": "ssr01_probe_tip_temp_c",           "value": 648.7,
     "unit": "°C", "robot_id": "SSR-01"},
    {"timestamp": "2026-09-01T06:00Z",
     "sensor": "ssr01_sample_volume_mL",           "value": 2.1,
     "unit": "mL", "robot_id": "SSR-01"},
    {"timestamp": "2026-09-01T06:00Z",
     "sensor": "ssr01_radiation_dose_mGy_h",       "value": 85.4,
     "unit": "mGy/h", "robot_id": "SSR-01"},
    {"timestamp": "2026-09-01T06:00Z",
     "sensor": "ssr01_containment_vessel_temp_c",  "value": 32.1,
     "unit": "°C", "robot_id": "SSR-01"},
], source_id="SSR-01-primary-sampling-20260901T06Z")

# Log sample analysis result
loader.ingest_text(
    rag,
    text=(
        "SSR-01 sample analysis result — primary salt sample 2026-09-01. "
        "Sample volume: 2.1 mL. Sampling location: primary loop hot leg. "
        "ICP-OES results: Cr = 8.2 ppm (nominal < 15 ppm), "
        "Fe = 3.1 ppm (nominal < 5 ppm), Ni = 1.4 ppm (nominal < 3 ppm). "
        "UF₃/UF₄ ratio: 0.012 (within target 0.010–0.020). "
        "Redox potential: -318 mV (nominal). "
        "No anomalous fission-product peaks detected. "
        "Training label: task_success=True, chemistry_nominal=True, "
        "corrosion_rate_acceptable=True."
    ),
    source_id="SSR-01-sample-result/primary-20260901",
    data_type="characterisation_report",
)
```

---

## 3 — Dataset Export for Foundation-Model Fine-Tuning

```python
# Export labelled sampling episodes for training the probe-insertion policy
answer = rag.answer(
    "List all stored SSR-01 sampling episodes. For each provide: "
    "sampling location, probe insertion depth, sample volume achieved, "
    "probe tip temperature, key chemistry results (Cr, UF₃/UF₄), and "
    "training labels (task_success, chemistry_nominal). "
    "Flag any episodes where sample volume was outside 1.5–3 mL target range."
)
print(answer)
```

---

## Foundation-Model Training Summary

| Data stream | Source | Data-layer API | FM use |
|---|---|---|---|
| ORNL sampling procedures & chemistry ranges | ORNL archive | `rag.answer()` | Pre-training / RAG context |
| Probe depth, temperature, sample volume | SSR-01 sensors | `ingest_sensor_snapshot()` | State representation |
| ICP-OES / redox analysis results | Characterisation records | `ingest_text()` (characterisation_report) | Supervised fine-tuning |
| Episode outcome labels | Episode logs | `ingest_text()` (event_log) | RL reward signal |
