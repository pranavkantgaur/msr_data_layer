# How the MSR Data Layer Assists the Koger (1972) Hastelloy N Corrosion Study

> **Report:** Koger J.W. — *"Corrosion and Mass Transfer Characteristics of
> Some Molten Fluorides in Hastelloy N"*, ORNL-TM-4273, Oak Ridge National
> Laboratory, 1972.
>
> One of the most comprehensive experimental datasets on structural-material
> compatibility with molten fluoride salts, covering **static capsule tests**
> and **forced-convection thermal-gradient loop tests** of **Hastelloy N
> (INOR-8)** and modified variants in FLiBe, FLiNaK, and LiF-ThF₄ at
> **550–750 °C** for durations up to **25 000 h**.
>
> Key measurements include: mass change per coupon vs. position (hot zone /
> cold zone), SEM/optical cross-section analysis, chemical analysis of salt
> samples (dissolved Cr, Fe, Mo, Ni), and post-exposure mechanical testing.
> The report establishes the **~1 mg/(dm²·month)** corrosion-rate benchmark
> for Hastelloy N in clean FLiBe that is cited in virtually every subsequent
> MSR materials paper.

---

## 1 — Design Phase: Retrieving Loop-Test Geometry and Baseline Conditions

**Paper connection:** Koger's loop tests used carefully controlled temperature
gradients (ΔT = 100–150 °C between hot and cold legs) to drive mass transfer.
The report documents exact loop dimensions, coupon positions relative to
hot-leg midpoint, flow velocity, and salt composition for each run.  Modern
researchers replicating or extending these tests need this parameter set as a
traceable baseline.

**Data-layer capability:** Query the ORNL OCR archive to retrieve loop
geometry and baseline corrosion parameters before designing new tests.

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

answer = rag.answer(
    "What were the loop geometry, temperature gradient, flow velocity, "
    "and salt composition used in Koger's Hastelloy N thermal-convection "
    "loop tests in FLiBe? Include the ORNL-TM report number and coupon "
    "positions relative to the hot-leg maximum."
)
print(answer)
```

---

## 2 — During Test: Logging Loop Thermal Conditions

**Paper connection:** Koger's thermal-convection loops ran at hot-leg
temperatures of 680–750 °C with cold-leg temperatures 100–150 °C lower.
Temperature stability over months-long tests is critical: any excursion
changes the thermodynamic driving force for mass transfer and invalidates the
cumulative mass-change record.

**Data-layer capability:** Periodic thermocouple readings are stored as sensor
snapshots linked to a specific loop-run identifier.

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "1971-06-01T08:00Z",
     "sensor": "hot_leg_max_temp_c",       "value": 704.8, "unit": "°C",
     "loop_run": "TL-FLiBe-04"},
    {"timestamp": "1971-06-01T08:00Z",
     "sensor": "cold_leg_min_temp_c",      "value": 591.3, "unit": "°C",
     "loop_run": "TL-FLiBe-04"},
    {"timestamp": "1971-06-01T08:00Z",
     "sensor": "salt_flow_velocity_cms",   "value": 8.2,   "unit": "cm/s",
     "loop_run": "TL-FLiBe-04"},
], source_id="TL-FLiBe-04-1971-06-01T08Z")
```

A query like *"Were there any periods during loop TL-FLiBe-04 where the hot-leg
temperature deviated more than ±5 °C from the target 705 °C set-point, and
for how long?"* is answerable from the ingested sensor history.

---

## 3 — Post-Test: Storing Coupon Mass-Change Records

**Paper connection:** The central result of Koger's work is the mass-change
profile along the loop: **material loss at the hot end** and **mass gain at the
cold end**, confirming thermodynamic dissolution in the hot zone and
re-deposition in the cold zone.  Each coupon's position, exposure duration,
mass before and after, and surface area are tabulated in the report.

**Data-layer capability:** Each coupon result is ingested as a structured
characterisation record with position metadata, enabling hot-zone vs.
cold-zone comparisons.

```python
# Hot-leg coupon (position P-3, ~700 °C)
loader.ingest_text(
    rag,
    text=(
        "Mass change — Hastelloy N coupon P-3 (loop TL-FLiBe-04, hot leg, "
        "position 25 cm from hot-leg midpoint, nominal 700 °C, 10000 h). "
        "Pre-test mass: 18.342 g. Post-test mass: 18.211 g. "
        "Mass loss: 131 mg. Surface area: 12.8 cm². "
        "Specific mass loss: 10.2 mg/cm². "
        "Corrosion rate: 0.87 mg/(dm²·month)."
    ),
    source_id="mass-change/TL-FLiBe-04/P-3/10000h",
    data_type="characterisation_report",
)

# Cold-leg coupon (position P-12, ~600 °C)
loader.ingest_text(
    rag,
    text=(
        "Mass change — Hastelloy N coupon P-12 (loop TL-FLiBe-04, cold leg, "
        "position 15 cm from cold-leg midpoint, nominal 600 °C, 10000 h). "
        "Pre-test mass: 18.290 g. Post-test mass: 18.351 g. "
        "Mass gain: 61 mg. Surface area: 12.8 cm². "
        "Specific mass gain: 4.8 mg/cm² (deposit from hot-leg dissolution)."
    ),
    source_id="mass-change/TL-FLiBe-04/P-12/10000h",
    data_type="characterisation_report",
)
```

---

## 4 — Post-Test: Ingesting Salt Chemistry Analysis

**Paper connection:** Koger periodically sampled the loop salt to monitor
**dissolved Cr, Fe, Mo, Ni** concentrations, demonstrating that chromium is the
preferentially leached element (highest thermodynamic driving force for
dissolution) and that its concentration in the salt increased over time before
reaching a steady state consistent with the hot-zone depletion rate.

**Data-layer capability:** Salt sample results are ingested per run per
time-point, enabling dissolved-metal trend analysis across all loop runs.

```python
loader.ingest_text(
    rag,
    text=(
        "Salt chemistry sample — loop TL-FLiBe-04, 10000 h time-point. "
        "ICP analysis: Cr 18 ppm, Fe 3 ppm, Mo 1.2 ppm, Ni 2.1 ppm. "
        "UF₃/UF₄ not applicable (no uranium in this coolant-salt loop). "
        "Sample collected: 1972-03-10. Analyst: J. Koger / ORNL Analytical Div."
    ),
    source_id="salt-chemistry/TL-FLiBe-04/10000h",
    data_type="characterisation_report",
)
```

---

## 5 — Post-Test: Storing SEM Cross-Section and Elemental-Profile Data

**Paper connection:** SEM/EDS analysis of Koger's hot-leg coupons revealed
a **Cr-depleted subsurface zone** ~10–30 µm deep after 10 000 h, consistent
with solid-state diffusion of Cr to the surface followed by fluoride dissolution.
Cold-leg coupons showed corresponding Cr-enriched surface deposits from
re-precipitation.

**Data-layer capability:** SEM/EDS profiles are stored with coupon and position
metadata, enabling comparisons across alloys and salt compositions.

```python
loader.ingest_text(
    rag,
    text=(
        "SEM/EDS cross-section — Hastelloy N coupon P-3 (loop TL-FLiBe-04, "
        "hot leg, 700 °C, 10000 h). "
        "Cr-depleted zone depth: 22 µm (EDS line scan, threshold: 50% of "
        "bulk Cr 7 wt%). "
        "Mo not depleted. Ni slightly depleted (<2 µm). "
        "No grain-boundary attack observed (consistent with clean FLiBe). "
        "Surface layer: thin (~1 µm) Cr-deficient oxide."
    ),
    source_id="sem-eds/TL-FLiBe-04/P-3/10000h",
    data_type="characterisation_report",
)
```

---

## 6 — Cross-Run Analysis: Comparing Alloy Variants and Salt Compositions

**Paper connection:** Koger tested both **standard Hastelloy N** and
**Ti-modified Hastelloy N** (improved tellurium-embrittlement resistance)
in several salt systems (FLiBe, FLiNaK, LiF-ThF₄), allowing direct
alloy-vs-salt comparison at matched temperatures and durations.

**Data-layer capability:** Once all coupon records are ingested across all
loop runs, the RAG pipeline synthesises cross-alloy and cross-salt comparisons.

```python
answer = rag.answer(
    "Compare the Cr-depletion depths and specific mass losses measured for "
    "standard Hastelloy N vs. Ti-modified Hastelloy N in FLiBe loop tests "
    "at 700 °C and 10000 h. Does Ti modification reduce the corrosion rate?"
)

answer = rag.answer(
    "At equivalent temperature and duration, how do the Hastelloy N corrosion "
    "rates compare between FLiBe, FLiNaK, and LiF-ThF4? What salt chemistry "
    "factor explains the differences?"
)
```

---

## 7 — Connecting to Modern Corrosion Studies

**Paper connection:** Koger's benchmark rates (~1 mg/(dm²·month) for Hastelloy N
in FLiBe at 700 °C) are the standard reference against which all subsequent
alloy-corrosion papers — including Lucas et al. (2025) on 316L SS — position
their results.

**Data-layer capability:** With both Koger's ORNL data and newer papers in the
same knowledge base, the RAG pipeline can directly compare materials and
benchmark new results.

```python
answer = rag.answer(
    "How does the corrosion rate of 316L stainless steel in purified FLiNaK "
    "at 600 °C (from recent studies) compare to Koger's Hastelloy N benchmark "
    "of ~1 mg/(dm²·month) in FLiBe at 700 °C? What alloy and salt differences "
    "explain any discrepancy?"
)
```

---

## Summary: Koger (1972) × Data-Layer Capability

| Experimental phase | Data ingested | Data-layer capability |
|---|---|---|
| Design — loop geometry baseline | ORNL-TM-4273 parameters | `rag.load_msr_archive()` |
| During — loop thermal conditions | Hot/cold-leg temps, flow rate | `loader.ingest_sensor_snapshot()` |
| Post-test — coupon mass change | Per-coupon mass loss/gain, position | `loader.ingest_text()` (characterisation_report) |
| Post-test — salt chemistry | Dissolved Cr/Fe/Mo/Ni vs. time | `loader.ingest_text()` (characterisation_report) |
| Post-test — SEM/EDS profiles | Depletion-zone depth, surface layer | `loader.ingest_text()` (characterisation_report) |
| Cross-run — alloy / salt comparison | All loop runs, all alloys, all salts | `rag.answer()` |
| Benchmark — comparison with modern data | Koger + recent corrosion papers | `rag.answer()` |
