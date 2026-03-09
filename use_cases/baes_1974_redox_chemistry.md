# How the MSR Data Layer Assists the Baes (1974) Redox Chemistry Study

> **Paper:** Baes C.F. Jr. — *"The Chemistry and Thermodynamics of Molten
> Salt Reactor Fuels"*, *Journal of Nuclear Materials*, Vol. 51, No. 1,
> pp. 149–162, 1974.
>
> The foundational thermodynamic analysis of **redox potential control** in
> LiF-BeF₂-based fuel salt, establishing the quantitative relationship between
> the **UF₃/UF₄ ratio**, **corrosion of structural metals**, and **fission-product
> speciation**.  Experimental contributions include:
> * Electrochemical cell measurements (emf) for UF₄ + ½Be → UF₃ + ½BeF₂
>   equilibrium at 600–800 °C
> * Partition coefficients for Cr, Fe, Ni, Mo between salt and metal phase
>   as a function of redox potential
> * Free-energy data for key fission-product fluorides (CsF, SrF₂, BaF₂,
>   TeF₄, NbF₅) enabling prediction of speciation in the salt
> * Correlation of measured Cr dissolution rate with UF₃/UF₄ ratio across
>   MSRE operating history
>
> Baes (1974) established the still-current design rule: **maintain
> UF₃/U(total) ≥ 0.01** to prevent significant Cr and Ni dissolution from
> Hastelloy N.

---

## 1 — Design Phase: Retrieving Thermodynamic Data for Salt-Metal Reactions

**Paper connection:** Reactor designers and materials engineers need free-energy
data for the reactions that determine whether a structural metal will dissolve
in or be protected by the fuel salt.  Baes (1974) tabulated these values;
subsequent work added fission-product reactions.

**Data-layer capability:** Load the ORNL archive and query for
thermodynamic data relevant to the specific alloy-salt combination of interest.

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

answer = rag.answer(
    "What is the standard free energy of reaction for Cr + 2UF₄ → CrF₂ + 2UF₃ "
    "in FLiBe fuel salt at 650 °C and 700 °C? "
    "Provide the equilibrium constant and the corresponding equilibrium "
    "UF₃/UF₄ ratio below which Cr dissolution is thermodynamically spontaneous. "
    "Cite Baes 1974 or ORNL thermodynamic report numbers."
)
print(answer)
```

---

## 2 — Design Phase: Computing Speciation of Fission Products

**Paper connection:** Baes computed the stable chemical form of each fission
product in the fuel salt at various redox potentials — for example, whether
caesium exists as CsF (dissolved, no corrosion concern) or as Cs metal
(potentially deposits on cold surfaces), and whether tellurium exists as TeF₄
(salt-soluble, corrosion concern for Hastelloy N) or Te° (solid deposit).

**Data-layer capability:** Query the archive for fission-product speciation
data, combining Baes thermodynamics with ORNL radiochemistry measurements.

```python
answer = rag.answer(
    "According to Baes (1974) and subsequent ORNL work, in what chemical form "
    "does tellurium exist in FLiBe fuel salt at UF₃/UF₄ = 0.01 and 0.05? "
    "At what redox potential does Te speciate from TeF4 to Te° metal? "
    "Why does this matter for Hastelloy N embrittlement?"
)

answer = rag.answer(
    "Which noble-metal fission products (Ru, Rh, Pd, Mo, Tc) are predicted "
    "by Baes thermodynamics to deposit as metals rather than dissolve as "
    "fluorides at MSRE operating conditions (UF₃/UF₄ = 0.01–0.05, 650 °C)?"
)
```

---

## 3 — During Operations: Monitoring Redox Potential in Real Time

**Paper connection:** Baes showed that the electrochemical cell voltage of a
Pt electrode vs. a Ni/NiF₂ reference in the salt is a direct measure of the
fluoride ion activity, which in turn determines the UF₃/UF₄ ratio.  This
measurement was implemented in the MSRE and is the basis for all proposed
online redox monitoring in modern MSR designs.

**Data-layer capability:** Online redox-sensor readings are ingested as
continuous sensor snapshots, enabling detection of excursions below the
corrosion-threshold.

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log electrochemical cell voltage every 15 min
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2025-06-01T12:00Z",
     "sensor": "redox_emf_mV",           "value": -312.4, "unit": "mV",
     "reference": "Ni/NiF₂", "salt": "FLiBe-UF₄"},
    {"timestamp": "2025-06-01T12:00Z",
     "sensor": "uf3_uf4_ratio_calc",     "value": 0.021, "unit": "mol/mol",
     "derived_from": "redox_emf_mV",
     "thermodynamic_basis": "Baes_1974_Eq14"},
    {"timestamp": "2025-06-01T12:00Z",
     "sensor": "fuel_salt_temp_c",       "value": 651.2, "unit": "°C"},
], source_id="redox-monitor-2025-06-01T12Z")
```

Once ingested, a query like *"Over the last 30 days of operation, how many
times did the UF₃/UF₄ ratio fall below 0.010, for how long, and what was the
salt temperature during those excursions?"* is answerable from the stored
time-series.

---

## 4 — During Operations: Logging Reductant Additions

**Paper connection:** When the UF₃/UF₄ ratio drops below the target (oxidising
excursion), the standard remedy is to add beryllium metal to convert UF₄ to
UF₃ (Be + 2UF₄ → BeF₂ + 2UF₃).  The amount of Be required is calculable from
the Baes thermodynamic data and the salt volume.

**Data-layer capability:** Each Be-addition event is recorded as an event log,
enabling audit of how frequently reducing conditions needed restoration and
whether they correlated with specific operational modes.

```python
loader.ingest_text(
    rag,
    text=(
        "Redox-restoration event — 2025-06-03T08:00Z. "
        "Trigger: UF₃/UF₄ ratio fell to 0.007 (redox_emf = -288 mV) "
        "at 18:00Z on 2025-06-02, below the lower alarm threshold of 0.010. "
        "Duration below threshold: 14 h. "
        "Corrective action: 1.2 g Be metal added to fuel salt at 08:00Z. "
        "Post-addition measurement at 10:00Z: UF₃/UF₄ = 0.023 (restored). "
        "Operator: J. Singh. Approved by: Chemistry Supervisor."
    ),
    source_id="event/redox-restoration-20250603",
    data_type="event_log",
)
```

---

## 5 — Post-Campaign: Correlating Redox History with Corrosion Measurements

**Paper connection:** Baes (1974) showed that the cumulative Cr dissolution
from the MSRE primary circuit was quantitatively consistent with the integrated
time spent at low UF₃/UF₄ ratios.  This correlation was validated against
periodic ICP measurements of the fuel salt.

**Data-layer capability:** Once sensor history (UF₃/UF₄ time-series) and
characterisation records (salt ICP-OES) are both ingested, the RAG pipeline
can reproduce this correlation for any operating campaign.

```python
loader.ingest_text(
    rag,
    text=(
        "ICP-OES salt analysis — campaign Q2-2025, 1500 h time-point. "
        "Dissolved metals in FLiBe fuel salt: "
        "Cr 6.8 ppm, Fe 1.2 ppm, Ni 0.9 ppm, Mo 0.3 ppm. "
        "Cumulative time with UF₃/UF₄ < 0.010 during Q2-2025: 38 h. "
        "Analysis date: 2025-07-01. Lab: Analytical Chemistry, Site A."
    ),
    source_id="icp-oes/campaign-Q2-2025/1500h",
    data_type="characterisation_report",
)

answer = rag.answer(
    "Using the Baes (1974) thermodynamic correlation and the ICP-OES salt "
    "analyses for campaign Q2-2025, estimate the total Cr mass leached from "
    "Hastelloy N primary circuit surfaces. How does this compare to the MSRE "
    "benchmark of ~1 mg/(dm²·month) at nominal operating conditions?"
)
```

---

## 6 — Designing Modern Electrochemical Monitoring Systems

**Paper connection:** Baes' electrochemical cell design used a Pt electrode
immersed in the salt against a Ni/NiF₂ reference.  Modern MSR programmes are
developing improved sensor geometries (W electrode, solid-electrolyte probes)
calibrated against the same Baes thermodynamic framework.

**Data-layer capability:** The knowledge base bridges ORNL sensor designs and
modern literature, enabling calibration curve development and uncertainty
quantification.

```python
import subprocess
subprocess.run(["python", "msr_kb_sources.py", "--update-openalex"])

answer = rag.answer(
    "What electrochemical sensor designs have been proposed or demonstrated "
    "since 2010 for online measurement of the UF₃/UF₄ ratio (or equivalent "
    "redox potential) in molten fluoride salts? How are they calibrated against "
    "Baes (1974) thermodynamic data? What uncertainties are reported?"
)
```

---

## Summary: Baes (1974) × Data-Layer Capability

| Experimental / operational phase | Data ingested | Data-layer capability |
|---|---|---|
| Design — thermodynamic data retrieval | Baes free-energy tables, ORNL archive | `rag.load_msr_archive()` + `rag.answer()` |
| Design — fission-product speciation | Partition coefficients, speciation diagrams | `rag.answer()` |
| During — online redox monitoring | Electrochemical cell voltage, UF₃/UF₄ ratio | `loader.ingest_sensor_snapshot()` |
| During — reductant-addition events | Be-metal additions, pre/post redox values | `loader.ingest_text()` (event_log) |
| Post-campaign — Cr dissolution correlation | ICP-OES + UF₃/UF₄ time-series | `rag.answer()` over combined records |
| Sensor development — modern designs | OpenAlex electrochemistry papers | `rag.answer()` over ORNL + OpenAlex |
