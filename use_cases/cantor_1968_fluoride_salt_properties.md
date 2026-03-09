# How the MSR Data Layer Assists the Cantor et al. (1968) Fluoride Salt Properties Study

> **Report:** Cantor S. (editor) — *"Physical Properties of Molten-Salt Reactor
> Fuel, Coolant, and Flush Salts"*, ORNL-4229, Oak Ridge National Laboratory,
> August 1968.  Contributing authors: Cantor S., Cooke J.W., Dworkin A.S.,
> Robbins G.D., Thoma R.E., Watson G.M.
>
> The definitive experimental compendium of thermophysical and transport
> properties of the molten fluoride salts considered for the MSR programme:
> **FLiBe (LiF-BeF₂)**, **FLiNaK (LiF-NaF-KF)**, **flush salt (NaF-ZrF₄)**,
> and **fuel salt (LiF-BeF₂-ThF₄-UF₄)**.  Measurements cover the temperature
> range **450–900 °C** and include:
> * Density (pycnometry / dilatometry)
> * Dynamic viscosity (rotating-cylinder viscometer)
> * Thermal conductivity (transient hot-wire and laser-flash)
> * Heat capacity (drop calorimetry and DSC)
> * Liquidus / solidus temperatures (thermal-arrest / DSC)
> * Electrical conductivity (impedance spectroscopy)
>
> ORNL-4229 is cited in essentially every paper on molten-fluoride-salt reactor
> thermal hydraulics, safety analysis, and chemistry.

---

## 1 — Design Phase: Retrieving Property Data for a Specific Salt Composition

**Paper connection:** A reactor thermal-hydraulics engineer designing a new
FLiBe coolant loop needs the density, viscosity, and thermal conductivity of
the exact composition being considered (e.g., LiF 67 mol% – BeF₂ 33 mol%)
at the operating temperature range.  ORNL-4229 contains these values as
tabulated data and polynomial fits.

**Data-layer capability:** Query the ORNL archive for specific property values
or fitting coefficients.

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

answer = rag.answer(
    "What is the density of FLiBe (LiF-BeF₂ eutectic, 67-33 mol%) as a "
    "function of temperature between 500 and 750 °C? "
    "Provide the linear-fit coefficients (rho = a + b*T) and the source "
    "ORNL report number."
)
print(answer)

answer = rag.answer(
    "What is the dynamic viscosity of FLiNaK at 600 °C and 700 °C? "
    "Include measurement uncertainty and the experimental method used."
)
```

---

## 2 — Design Phase: Comparing Properties Across Multiple Salt Candidates

**Paper connection:** MSR designers routinely need to compare FLiBe, FLiNaK,
and chloride salts on multiple property axes simultaneously — particularly
thermal conductivity (affects heat-transfer coefficient) and viscosity
(affects pump work and natural-circulation viability).

**Data-layer capability:** A single RAG query retrieves and compares property
tables from across the archive and recent papers.

```python
answer = rag.answer(
    "Compare the thermal conductivity and kinematic viscosity of FLiBe, "
    "FLiNaK, and NaF-ZrF4 at 650 °C. Which salt has the best heat-transfer "
    "characteristics for a natural-circulation loop? "
    "Cite ORNL-4229 data where available."
)
```

---

## 3 — During Experiment: Logging Property Measurement Conditions

**Paper connection:** ORNL-4229's property measurements required precise
control of sample temperature, inert atmosphere, and salt composition.
The measurement protocols — especially for viscosity (rotating-cylinder
method) and thermal conductivity (transient hot-wire) — involve multiple
calibration steps and atmosphere control that must be logged for traceability.

**Data-layer capability:** Measurement-run conditions are stored as sensor
snapshots linking them to specific property results.

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log conditions for each viscosity measurement point
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "1967-04-12T10:00Z",
     "sensor": "sample_temperature_c",     "value": 601.8, "unit": "°C",
     "property": "dynamic_viscosity",
     "salt_id": "FLiNaK-batch-007"},
    {"timestamp": "1967-04-12T10:00Z",
     "sensor": "furnace_atmosphere",       "value": "Ar", "unit": "—",
     "salt_id": "FLiNaK-batch-007"},
    {"timestamp": "1967-04-12T10:00Z",
     "sensor": "o2_impurity_ppm",          "value": 3.1,  "unit": "ppm",
     "salt_id": "FLiNaK-batch-007"},
], source_id="viscosity-FLiNaK-batch-007-600C-1967-04-12")
```

---

## 4 — Post-Measurement: Ingesting Tabulated Property Results

**Paper connection:** ORNL-4229 contains extensive data tables: density vs.
temperature for 15+ compositions, viscosity at 10–15 temperature points per
salt, heat-capacity values at 50 °C intervals.  Re-ingesting this structured
data enables programmatic retrieval without format-conversion overhead.

**Data-layer capability:** Each property table is ingested as a structured
characterisation record, one entry per temperature point or composition.

```python
# Ingest density data for FLiBe at a single temperature
loader.ingest_text(
    rag,
    text=(
        "Density measurement — FLiBe (LiF 66 mol% – BeF₂ 34 mol%), 600 °C. "
        "Method: Archimedean pycnometry. "
        "Measured density: 2.006 g/cm³. "
        "Linear fit (450–750 °C): rho = 2.280 - 0.000490*T (g/cm³, T in °C). "
        "Source: ORNL-4229, Table 2.1. Measurement uncertainty: ±0.3%."
    ),
    source_id="property/FLiBe-66-34/density/600C",
    data_type="characterisation_report",
)

# Ingest viscosity data for FLiNaK
loader.ingest_text(
    rag,
    text=(
        "Dynamic viscosity — FLiNaK (LiF 46.5 – NaF 11.5 – KF 42 mol%), 600 °C. "
        "Method: rotating-cylinder viscometer (Hastelloy N spindle). "
        "Measured viscosity: 2.9 mPa·s (2.9 cP). "
        "Arrhenius fit (500–800 °C): ln(η) = -1.03 + 3624/T (T in K). "
        "Source: ORNL-4229, Table 4.3. Measurement uncertainty: ±5%."
    ),
    source_id="property/FLiNaK-eutectic/viscosity/600C",
    data_type="characterisation_report",
)
```

---

## 5 — Cross-Property Analysis: Deriving Thermal-Hydraulic Parameters

**Paper connection:** Reactor designers combine the measured properties into
derived parameters: Prandtl number (Pr = µ·Cp/k), thermal diffusivity
(α = k/(ρ·Cp)), and Nusselt number correlations.  ORNL-4229 provides the
constituent measurements; the derived parameters must be computed and checked
for self-consistency.

**Data-layer capability:** Once all property records are ingested, the RAG
pipeline can retrieve the constituent values and guide the calculation.

```python
answer = rag.answer(
    "Using the ORNL-4229 data for FLiBe, calculate the Prandtl number at 650 °C "
    "(Pr = dynamic viscosity × heat capacity / thermal conductivity). "
    "What heat-transfer coefficient would be expected for turbulent flow in a "
    "25 mm diameter tube at 2 m/s flow velocity?"
)
```

---

## 6 — Identifying Property Gaps for Modern Salt Candidates

**Paper connection:** ORNL-4229 covers the classic MSR salts but does not
include newer candidates: chloride eutectics (NaCl-MgCl₂), fluoride-nitrate
mixtures, or thorium-rich fluoride compositions at high ThF₄ loadings.
Modern researchers need to identify where ORNL data exists and where new
measurements are needed.

**Data-layer capability:** Combine archive data with OpenAlex literature to
map the property measurement landscape.

```python
import subprocess
subprocess.run(["python", "msr_kb_sources.py", "--update-openalex"])

answer = rag.answer(
    "Which thermophysical properties of NaCl-MgCl2 eutectic have been "
    "measured since 2000 and published in peer-reviewed journals? "
    "Where do gaps remain compared to the ORNL-4229 coverage of FLiBe? "
    "Focus on: density, viscosity, thermal conductivity, heat capacity."
)
```

---

## Summary: Cantor et al. (1968) × Data-Layer Capability

| Experimental phase | Data ingested | Data-layer capability |
|---|---|---|
| Design — property lookup | ORNL-4229 tables via RAG | `rag.load_msr_archive()` + `rag.answer()` |
| Design — cross-salt comparison | FLiBe vs. FLiNaK vs. NaF-ZrF₄ | `rag.answer()` |
| During — measurement run conditions | Temperature, atmosphere, O₂ impurity | `loader.ingest_sensor_snapshot()` |
| Post-measurement — tabulated data | Density, viscosity, k, Cp per composition | `loader.ingest_text()` (characterisation_report) |
| Analysis — derived parameters | Pr, Re, Nu, heat-transfer coefficient | `rag.answer()` |
| Gap analysis — modern salt candidates | ORNL archive + OpenAlex property papers | `rag.answer()` over combined KB |
