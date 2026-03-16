# How the MSR Data Layer Supports the Yang et al. (2025) Molten Salt Pump Thermal Optimisation Study

> **Paper:** Yang J., Zhang J.-Y., Fu Y., Shen X.-C., Li Z.-J. — *"Optimization
> of Thermal Resistance for High-Temperature Molten Salt Pump Based on Response
> Surface Methodology and NSGA-II Algorithm"*, Shanghai Institute of Applied
> Physics, Chinese Academy of Sciences / University of Chinese Academy of
> Sciences, submitted July 2025, accepted September 2025.
> Supported by Gansu Major Scientific and Technological Special Project
> No. 23ZDGH001.
>
> The molten salt in the reactor primary loop operates at up to **650 °C**,
> but the core components of the pump's upper driving end — **magnetic
> levitation bearings** and the **motor** — must stay within a much lower
> temperature range to prevent material aging and shortened service life.
> The paper addresses this mismatch by:
>
> 1. Identifying the four structural parameters that most strongly control
>    axial heat conduction from the salt to the driving end:
>    * **δ₁** — outer cylinder wall thickness (20–50 mm)
>    * **δ₂** — outer insulation-screen ring-frame thickness (8.7–17.1 mm)
>    * **δ₃** — inner insulation-screen ring-frame thickness (5.7–10.8 mm)
>    * **D** — shaft diameter (80–160 mm)
> 2. Ranking their importance by Analytic Hierarchy Process (AHP):
>    δ₁ 49.98 % · D 31.95 % · δ₂ 11.91 % · δ₃ 6.15 %.
> 3. Building a quadratic RSM (Box-Behnken Design, 27 runs) approximation
>    model for two competing objectives — magnetic-bearing temperature **Y₁**
>    and rotor bending-stiffness index **Y₂** (R² = 0.9789).
> 4. Applying NSGA-II (population 100, 500 generations) to generate the
>    Pareto front.
>
> Optimal design:
> * δ₁ = 22.45 mm · δ₂ = 8.7 mm · δ₃ = 10.41 mm · D = 160 mm
> * Predicted bearing temperature: 155.4 °C → CFD-verified: 158.7 °C (2.12 % error)
> * Temperature reduction vs. original design: **10.3 °C**

---

## 1 — Design Phase: Retrieving ORNL Baseline Data for Pump Materials and Salt Properties

**Paper connection:** The pump structural components are fabricated from
**316H stainless steel** (λ = 14.1 W/(m·K), Cp = 452 J/(kg·K)) operating
in contact with **LiF-BeF₂-ZrF₄ coolant salt at 650 °C**.  Before setting
thermal boundary conditions and material property inputs for ANSYS Fluent,
designers need the best available measurements of thermal conductivity,
specific heat, and viscosity for the fluoride salts, and high-temperature
mechanical properties of 316H in a fluoride environment.

**Data-layer capability:** Load the ORNL archive and query it for the
relevant thermal and thermophysical data.

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()   # one-time; re-runs add only new files

# Retrieve thermal properties of FLiBe / NaF-BeF₂ relevant to pump design
answer = rag.answer(
    "What are the thermal conductivity, specific heat, viscosity, and density "
    "of FLiBe (Li₂BeF₄) and NaF-BeF₂ fluoride salts at 600–700 °C as "
    "measured in ORNL experiments? Include report numbers and temperature "
    "dependence if available."
)
print(answer)

# Retrieve 316H / austenitic steel high-temperature thermal data from ORNL
answer = rag.answer(
    "What high-temperature mechanical and thermal properties (yield strength, "
    "thermal conductivity, creep resistance) of 316 stainless steel or INOR-8 "
    "in fluoride salt service at 600–700 °C are documented in ORNL MSR reports? "
    "Are there fatigue or aging data for 316H in prolonged contact with "
    "molten fluorides?"
)
print(answer)

# Retrieve ORNL pump thermal experience
answer = rag.answer(
    "What thermal management challenges and solutions were documented for "
    "the molten salt pump bearings and mechanical seals in the MSRE primary "
    "circuit? What bearing temperatures were measured, and what insulation "
    "or heat-shield arrangements were used to protect the pump driving end?"
)
print(answer)
```

```bash
python msr_digital_twin_with_rag.py \
  "Summarise ORNL molten salt pump design data: bearings, shaft seals, "
  "thermal isolation between salt and drive end"
```

---

## 2 — Simulation Data Storage: Ingesting the 27 BBD Experimental Design Runs

**Paper connection:** The Box-Behnken Design (BBD) generates 27 structured
combinations of the four parameters (δ₁, δ₂, δ₃, D).  Each combination is
evaluated by ANSYS Fluent to obtain the magnetic-bearing temperature Y₁ and
the bending-stiffness index Y₂.  These 27 simulation records are the primary
training data for the RSM model; they must be stored traceably so that the
RSM fit can be reproduced and audited.

**Data-layer capability:** Each simulation run is ingested as a
`characterisation_report` record with full parameter and result metadata.

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Example: ingest four representative BBD runs (full programme: 27 records)
bbd_runs = [
    # (run_id, δ1, δ2, δ3, D, T_bearing_degC, stiffness_index)
    ("BBD-01",  20.0,  8.7, 8.25, 120, 150.0, 0.812),
    ("BBD-02",  50.0,  8.7, 8.25, 120, 165.7, 0.891),
    ("BBD-03",  20.0, 17.1, 8.25, 120, 156.1, 0.803),
    ("BBD-04",  50.0, 17.1, 8.25, 120, 169.5, 0.896),
    # ... remaining 23 runs ingested similarly
]

for run_id, d1, d2, d3, D, T, K in bbd_runs:
    loader.ingest_text(
        rag,
        text=(
            f"CFD simulation record — Box-Behnken Design run {run_id}. "
            f"Source: Yang et al. (2025), Appendix A / Table 3. "
            f"Tool: ANSYS Fluent, mesh: 2.06 M cells. "
            f"Design variables: "
            f"δ₁ (outer cylinder thickness) = {d1} mm, "
            f"δ₂ (outer insulation frame) = {d2} mm, "
            f"δ₃ (inner insulation frame) = {d3} mm, "
            f"D (shaft diameter) = {D} mm. "
            f"Responses: "
            f"Y₁ (magnetic bearing avg temperature) = {T} °C, "
            f"Y₂ (bending stiffness index) = {K}. "
            f"Boundary conditions: salt region 650 °C, upper shell 100 °C, "
            f"outer cylinder ambient 200 °C. "
            f"Material: 316H SS (λ = 14.1 W/(m·K)). "
            f"Insulation: nano-ceramic (λ = 0.035 W/(m·K))."
        ),
        source_id=f"cfd/yang-2025/BBD/{run_id}",
        data_type="characterisation_report",
    )

# Verify all runs are stored and retrieve summary statistics
answer = rag.answer(
    "List all ingested CFD simulation runs from the Yang et al. (2025) "
    "Box-Behnken Design study. For each run provide: run ID, parameter values "
    "(δ₁, δ₂, δ₃, D), predicted bearing temperature, and stiffness index. "
    "What is the range of bearing temperatures across all runs?"
)
print(answer)
```

---

## 3 — RSM Model Storage: Preserving the Quadratic Approximation for Future Queries

**Paper connection:** The paper derives a quadratic polynomial RSM equation
relating the four design variables to the bearing temperature Y₁
(R² = 0.9789, R²_adj = 0.9920).  This model is the key transferable output
of the study — it allows rapid design-space exploration without running new
CFD cases.  Storing it in the data layer makes it queryable by future
designers and auditors.

**Data-layer capability:** The RSM model equation, its coefficients, and its
validation metrics are ingested as a structured record.

```python
loader.ingest_text(
    rag,
    text=(
        "RSM approximation model — Yang et al. (2025) molten salt pump "
        "bearing temperature. "
        "Response: Y₁ = average temperature of magnetic levitation bearing (°C). "
        "Variables: δ₁ (mm), δ₂ (mm), δ₃ (mm), D (mm). "
        "Model type: quadratic polynomial (Box-Behnken Design, 27 runs). "
        "Equation: "
        "T = 114.24397 "
        "+ 0.963708·δ₁ + 1.6544·δ₂ − 0.063204·δ₃ + 0.040636·D "
        "− 0.009127·δ₁·δ₂ − 0.001307·δ₁·δ₃ − 0.00025·δ₁·D "
        "− 0.011671·δ₂·δ₃ − 0.000893·δ₂·D + 0.001471·δ₃·D "
        "− 0.004019·δ₁² − 0.020786·δ₂² + 0.014738·δ₃² + 0.000161·D². "
        "Fit statistics: R² = 0.9789, R²_adj = 0.9920 (both > 0.9 threshold). "
        "AHP parameter weights: δ₁ 49.98%, D 31.95%, δ₂ 11.91%, δ₃ 6.15%."
    ),
    source_id="rsm-model/yang-2025/bearing-temp-quadratic",
    data_type="characterisation_report",
)

# Rapid design-space query using the stored RSM equation
answer = rag.answer(
    "Using the Yang et al. (2025) RSM model for molten salt pump bearing "
    "temperature, what bearing temperature is predicted for the following "
    "design point: δ₁ = 30 mm, δ₂ = 10 mm, δ₃ = 8 mm, D = 140 mm? "
    "Which parameter has the strongest influence on bearing temperature "
    "according to the AHP weights and RSM coefficients?"
)
print(answer)
```

---

## 4 — Optimisation History: Storing the NSGA-II Pareto Front

**Paper connection:** NSGA-II produces a Pareto front balancing two competing
objectives — minimising magnetic-bearing temperature Y₁ and maximising the
bending-stiffness index Y₂.  Storing the Pareto front and the selected
optimal point allows future design iterations to start from the documented
trade-off rather than re-running the optimisation.

**Data-layer capability:** The Pareto front solutions and the selected design
point are ingested with optimisation provenance metadata.

```python
# Store NSGA-II optimisation run metadata
loader.ingest_text(
    rag,
    text=(
        "NSGA-II optimisation run — Yang et al. (2025) molten salt pump. "
        "Algorithm: Non-dominated Sorting Genetic Algorithm II (Deb et al. 2002). "
        "Population size: 100. Generations: 500. Crossover probability: 0.9. "
        "Objectives: "
        "  Y₁ = minimise magnetic bearing temperature (°C), "
        "  Y₂ = maximise bending stiffness index (dimensionless). "
        "Design variable bounds: "
        "  δ₁ ∈ [20, 50] mm, δ₂ ∈ [8.7, 17.1] mm, "
        "  δ₃ ∈ [5.7, 10.8] mm, D ∈ [80, 160] mm. "
        "Selected Pareto-optimal design point: "
        "  δ₁ = 22.45 mm, δ₂ = 8.7 mm, δ₃ = 10.41 mm, D = 160 mm. "
        "RSM-predicted responses at optimal: Y₁ = 155.4 °C, Y₂ = 0.8575. "
        "CFD-verified temperature: 158.7 °C (error 2.12%). "
        "Temperature reduction vs. original design: 10.3 °C. "
        "Original design bearing temperature: ~169 °C."
    ),
    source_id="optim/nsga2/yang-2025/pareto-selected-design",
    data_type="characterisation_report",
)

# Query the stored optimisation result
answer = rag.answer(
    "What is the optimised structural geometry for the SINAP high-temperature "
    "molten salt pump driving end from Yang et al. (2025)? "
    "What bearing temperature reduction was achieved compared to the original "
    "design, and what was the CFD-verified error of the RSM prediction? "
    "Which structural parameter was most reduced in the optimal design?"
)
print(answer)
```

---

## 5 — Operational Monitoring: Real-Time Pump Driving-End Temperature Surveillance

**Paper connection:** The optimised design targets a magnetic-bearing
operating temperature of ~158.7 °C.  During actual reactor operation,
continuous temperature monitoring of the pump driving end is needed to
confirm that the optimised design performs as predicted and to give early
warning if bearing temperatures approach the design-life threshold.

**Data-layer capability:** `PlantDataLoader.ingest_sensor_snapshot()` logs
periodic pump health readings alongside the reactor operating state.

```python
# Log a routine pump health snapshot every 15 min during normal operation
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-06-15T10:00Z",
     "sensor": "pump_maglev_bearing_avg_temp_c",  "value": 161.2, "unit": "°C",
     "pump_id": "MSR-PUMP-001",
     "reactor_id": "LF-MSR-001",
     "design_limit_c": 175.0,
     "optimised_target_c": 158.7},
    {"timestamp": "2026-06-15T10:00Z",
     "sensor": "pump_maglev_bearing_max_temp_c",  "value": 163.8, "unit": "°C",
     "pump_id": "MSR-PUMP-001",
     "reactor_id": "LF-MSR-001"},
    {"timestamp": "2026-06-15T10:00Z",
     "sensor": "pump_shaft_temp_midpoint_c",       "value": 312.4, "unit": "°C",
     "pump_id": "MSR-PUMP-001",
     "reactor_id": "LF-MSR-001"},
    {"timestamp": "2026-06-15T10:00Z",
     "sensor": "pump_outer_cylinder_surface_temp_c","value": 198.6, "unit": "°C",
     "pump_id": "MSR-PUMP-001",
     "reactor_id": "LF-MSR-001"},
    {"timestamp": "2026-06-15T10:00Z",
     "sensor": "fuel_salt_pump_inlet_temp_c",      "value": 648.9, "unit": "°C",
     "pump_id": "MSR-PUMP-001",
     "reactor_id": "LF-MSR-001"},
    {"timestamp": "2026-06-15T10:00Z",
     "sensor": "reactor_power_pct",                "value": 98.5,  "unit": "%",
     "pump_id": "MSR-PUMP-001",
     "reactor_id": "LF-MSR-001"},
], source_id="LF-MSR-001-pump-health-2026-06-15T10Z")

# Long-term performance query
answer = rag.answer(
    "For pump MSR-PUMP-001 on reactor LF-MSR-001, how does the measured "
    "magnetic-bearing temperature compare to the optimised design target of "
    "158.7 °C from Yang et al. (2025)? Has the bearing temperature shown any "
    "upward drift over the logged operating history that could indicate "
    "insulation degradation?"
)
print(answer)
```

---

## 6 — Design Iteration Tracking: Comparing Multiple Pump Geometry Variants

**Paper connection:** The Yang et al. study evaluated one original pump
geometry and one NSGA-II-optimised geometry.  In a real development
programme, several intermediate variants and design revisions will exist.
The data layer provides a single queryable record of all geometry variants,
their predicted performances, and their CFD-verification results, preventing
duplication of simulation work across design iterations.

**Data-layer capability:** Each design variant is ingested with a versioned
`source_id`; queries can compare all variants side by side.

```python
# Ingest original (pre-optimisation) design as the baseline variant
loader.ingest_text(
    rag,
    text=(
        "Molten salt pump design variant — ORIGINAL (pre-optimisation). "
        "Source: Yang et al. (2025), baseline case. "
        "Parameters: δ₁ ≈ 35 mm (nominal), δ₂ ≈ 13 mm, δ₃ ≈ 8 mm, D ≈ 120 mm. "
        "CFD result: magnetic bearing avg temperature ≈ 169 °C. "
        "Bending stiffness index ≈ 0.85. "
        "Assessment: above optimal thermal target; basis for optimisation."
    ),
    source_id="pump-design/MSR-PUMP/v0-original-Yang2025",
    data_type="characterisation_report",
)

# Ingest the NSGA-II optimised design as variant v1
loader.ingest_text(
    rag,
    text=(
        "Molten salt pump design variant — NSGA-II OPTIMISED (v1). "
        "Source: Yang et al. (2025), Table 5 / Section 3.5. "
        "Parameters: δ₁ = 22.45 mm, δ₂ = 8.7 mm, δ₃ = 10.41 mm, D = 160 mm. "
        "RSM-predicted bearing temperature: 155.4 °C. "
        "CFD-verified bearing temperature: 158.7 °C (error 2.12%). "
        "Bending stiffness index: 0.8575. "
        "Temperature reduction vs. v0: 10.3 °C. "
        "Assessment: satisfies both thermal and structural requirements."
    ),
    source_id="pump-design/MSR-PUMP/v1-nsga2-optimised-Yang2025",
    data_type="characterisation_report",
)

# Compare all variants
answer = rag.answer(
    "List all ingested molten salt pump design variants for the SINAP LF-MSR "
    "pump. For each variant provide: design parameter values (δ₁, δ₂, δ₃, D), "
    "CFD-verified bearing temperature, bending stiffness index, and "
    "the temperature difference relative to the original design. "
    "Which variant achieves the lowest bearing temperature while maintaining "
    "acceptable structural stiffness?"
)
print(answer)
```

---

## 7 — Cross-Study Synthesis: Linking Pump Thermal Data to Materials and System Safety

**Paper connection:** The pump driving-end temperature directly affects the
service life of the magnetic levitation bearings — materials that degrade
faster at higher temperatures.  The Yang et al. optimisation reduces bearing
temperature by 10.3 °C.  Connecting this result to material-aging data for
bearing alloys and to pump-failure event records in the MSRE archive provides
the quantitative basis for a predictive maintenance schedule.

**Data-layer capability:** Once pump health data, materials data, and
historical pump failure/maintenance records are co-ingested, the RAG pipeline
can synthesise across all three sources to estimate remaining useful life.

```python
# Retrieve ORNL pump maintenance and failure data
answer = rag.answer(
    "What bearing failures, seal leaks, or mechanical problems were recorded "
    "for the primary fuel-salt pump during MSRE operation? What were the "
    "root causes, how many operating hours had elapsed, and what were the "
    "salt or temperature conditions at the time of failure? "
    "Cite ORNL-TM report numbers."
)
print(answer)

# Estimate maintenance interval impact of the 10.3 °C reduction
answer = rag.answer(
    "If magnetic levitation bearing service life follows an Arrhenius "
    "temperature-dependence (as documented in ORNL MSR pump reports or "
    "general bearing literature in the knowledge base), what fractional "
    "increase in bearing service life would a 10.3 °C reduction in operating "
    "temperature (from ~169 °C to ~158.7 °C) produce? "
    "Assume an activation energy consistent with the bearing material class."
)
print(answer)

# Connect pump data to system-level safety assessment
answer = rag.answer(
    "For the LF-MSR passive cooling analysis (Xue et al. 2026, stored in the "
    "data layer), the primary pump seizure accident is the most thermally "
    "severe scenario (peak fuel salt temperature 811.2 °C). Does the pump "
    "thermal optimisation from Yang et al. (2025) — specifically the increased "
    "shaft diameter D = 160 mm — affect the pump seizure torque or inertia "
    "parameters that would influence the transient temperature trajectory?"
)
print(answer)
```

---

## Summary: Yang et al. (2025) × Data-Layer Capability

| Research / operational phase | Data ingested | Data-layer capability |
|---|---|---|
| Pre-design materials retrieval | ORNL thermal properties of salts, 316H, pump components | `rag.answer()` over ORNL archive |
| CFD simulation database | 27 BBD runs — parameters, bearing temps, stiffness | `loader.ingest_text()` (characterisation_report) |
| RSM model preservation | Quadratic equation, coefficients, R² metrics | `loader.ingest_text()` (characterisation_report) |
| NSGA-II optimisation history | Pareto-selected design point and verification | `loader.ingest_text()` (characterisation_report) |
| Real-time pump health monitoring | Bearing temp, shaft temp, outer-cylinder temp | `loader.ingest_sensor_snapshot()` |
| Design variant comparison | Original vs. optimised geometry, CFD results | `loader.ingest_text()` + `rag.answer()` |
| Cross-study synthesis | Pump life, PRHRS accident coupling, ORNL failures | `rag.answer()` across multiple ingested sources |
