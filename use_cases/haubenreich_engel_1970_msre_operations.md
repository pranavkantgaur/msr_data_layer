# How the MSR Data Layer Assists the Haubenreich & Engel (1970) MSRE Study

> **Paper:** Haubenreich P.N. & Engel J.R. — *"Experience with the Molten
> Salt Reactor Experiment"*, *Nuclear Applications and Technology*, Vol. 8,
> pp. 118–136, 1970.
>
> One of the most widely cited papers in MSR history, reporting the complete
> operational experience of the **Molten Salt Reactor Experiment (MSRE)**:
> **13 172 effective full-power hours** of operation at ORNL between 1965 and
> 1969 using ²³⁵U-bearing FLiBe fuel salt at **650–700 °C**.
>
> The study covers: steady-state neutronics and temperature coefficients,
> online chemistry monitoring (UF₃/UF₄ redox ratio, fission-product
> behaviour), structural-material performance (INOR-8/Hastelloy N container),
> graphite moderator behaviour, and the detailed fuel-salt processing and
> clean-up operations that maintained salt purity throughout the experiment.

The sections below map each major experimental data stream from the MSRE to a
specific data-layer capability.

---

## 1 — Design Phase: Querying ORNL Archive for Pre-MSRE Design Basis

**Paper connection:** The MSRE was the culmination of a decade of materials,
chemistry, and reactor-physics studies.  Researchers building on the MSRE
experience need rapid access to the design-basis reports — loop corrosion
capsule data, salt-chemistry limits, Hastelloy N qualification tests — that
underpinned the reactor's construction.

**Data-layer capability:** Load and query the ORNL OCR archive.

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

answer = rag.answer(
    "What corrosion and mass-transfer data for INOR-8/Hastelloy N in FLiBe "
    "were used to set the MSRE structural-material design limits? "
    "Include ORNL report numbers, temperature range, and measured rates."
)
print(answer)
```

```bash
python msr_digital_twin_with_rag.py \
  "Summarise the pre-MSRE salt-chemistry control strategy for UF₃/UF₄ ratio"
```

---

## 2 — During Reactor Operation: Ingesting Continuous Process Sensor Data

**Paper connection:** The MSRE generated continuous time-series from dozens of
instruments: primary-loop thermocouples (hot-leg 663 °C, cold-leg 632 °C),
fuel-salt pump speed and differential pressure, secondary-salt flow, heat
exchanger inlet/outlet temperatures, and cover-gas pressure.  These readings
documented the reactor's power history and thermal transients.

**Data-layer capability:** `PlantDataLoader.ingest_sensor_snapshot()` stores
periodic instrument readings alongside the associated reactor state.

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log every 30 min from the plant DCS historian
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "1968-01-15T14:00Z",
     "sensor": "fuel_salt_hot_leg_temp_c",  "value": 663.4, "unit": "°C"},
    {"timestamp": "1968-01-15T14:00Z",
     "sensor": "fuel_salt_cold_leg_temp_c", "value": 632.1, "unit": "°C"},
    {"timestamp": "1968-01-15T14:00Z",
     "sensor": "reactor_power_mw",          "value": 7.34,  "unit": "MW(th)"},
    {"timestamp": "1968-01-15T14:00Z",
     "sensor": "fuel_pump_speed_rpm",       "value": 1175,  "unit": "rpm"},
], source_id="msre-ops-1968-01-15T14Z")
```

After ingestion, queries such as *"What was the average hot-leg temperature
during the ²³³U campaign and how did it compare to the ²³⁵U campaign?"* are
answerable from the stored time-series without manual log searches.

---

## 3 — During Operation: Logging Online Chemistry Measurements

**Paper connection:** The MSRE team measured the **UF₃/UF₄ ratio** (redox
potential indicator) continuously using a Pt/Ni reference electrode, and
periodically sampled the fuel salt for fission-product concentrations (Cs, Sr,
Ba, Nb, Mo, Ru, Rh, Pd, Ag, Te) as well as structural-metal dissolution
(Cr, Fe, Ni from INOR-8).

**Data-layer capability:** Chemistry readings are ingested as structured sensor
snapshots, enabling co-querying with process conditions.

```python
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "1968-01-15T14:00Z",
     "sensor": "uf3_uf4_ratio_mole_fraction",            "value": 0.012,  "unit": "mol/mol"},
    {"timestamp": "1968-01-15T14:00Z",
     "sensor": "dissolved_cr_ppm",         "value": 4.2,    "unit": "ppm"},
    {"timestamp": "1968-01-15T14:00Z",
     "sensor": "dissolved_fe_ppm",         "value": 1.1,    "unit": "ppm"},
    {"timestamp": "1968-01-15T14:00Z",
     "sensor": "xe135_cover_gas_pct",      "value": 0.32,   "unit": "%"},
], source_id="msre-chemistry-1968-01-15T14Z")
```

A subsequent query like *"What was the dissolved-Cr trend in the fuel salt
over the full ²³⁵U run, and did it correlate with UF₃/UF₄ excursions below
0.005?"* synthesises stored chemistry records and process history in one call.

---

## 4 — During Operation: Recording Maintenance Events and Fuel Additions

**Paper connection:** Throughout the MSRE's life, the team performed planned
and unplanned interventions: beryllium-metal additions to restore the redox
potential, UF₄ additions after fuel processing, He-sparging campaigns to
remove Kr and Xe, pump seal replacements, and heat-exchanger tube inspections.
These events must be correlated with chemistry and corrosion trends.

**Data-layer capability:** Event logs are ingested as free-text records so
they can be retrieved alongside time-series data.

```python
loader.ingest_text(
    rag,
    text=(
        "MSRE event log — 1968-03-22: Beryllium metal addition to fuel salt "
        "to restore reducing potential. Pre-addition UF₃/UF₄ ratio: 0.004 "
        "(oxidising; near design lower limit of 0.005). Post-addition target: "
        "UF₃/UF₄ ≥ 0.010. Be mass added: 0.8 g. Fuel salt volume: ~1800 L. "
        "Operator: J. Smith. Approved by chemistry lead: C.F. Baes."
    ),
    source_id="msre-event-19680322-be-addition",
    data_type="event_log",
)
```

After ingesting all events, a query like *"How many beryllium additions were
required per year, and was the frequency increasing or decreasing over the
reactor lifetime?"* is answerable from the event log.

---

## 5 — Post-Campaign: Storing Fission-Product Inventory Measurements

**Paper connection:** At campaign end, destructive analysis of graphite samples
and INOR-8 coupons (removed during planned maintenance windows) quantified the
**noble-metal fission-product deposits** (Ru, Rh, Pd, Mo) on graphite and metal
surfaces, and the **volatile fission-product** (Te, Cs, I) distribution
throughout the primary circuit.

**Data-layer capability:** Post-campaign characterisation results are ingested
per sample per location.

```python
loader.ingest_text(
    rag,
    text=(
        "MSRE graphite specimen GR-147 (location: core zone 2, position P-14). "
        "Removed at end of run-16 (1969-12-12). "
        "Noble-metal deposits (radiochemical analysis): "
        "Ru-106: 2.4 µCi/g, Rh-103m: 1.1 µCi/g, Pd-107: 0.8 µCi/g, "
        "Mo-99/Tc-99: 3.2 µCi/g. "
        "Te-132 surface deposit: 0.14 µg/cm². "
        "No visible corrosion of graphite matrix."
    ),
    source_id="msre-specimen/GR-147/end-of-run-16",
    data_type="characterisation_report",
)
```

---

## 6 — Post-Campaign: AI-Assisted Cross-Correlation via RAG

**Paper connection:** The Haubenreich & Engel paper synthesises 13 172 h of
operational data into conclusions about materials performance, chemistry
stability, and fission-product behaviour.  Future reactor designers need to
query this dataset to inform design choices for modern MSR programmes.

**Data-layer capability:** Once the full MSRE operational record is ingested,
the RAG pipeline enables complex cross-dataset queries.

```python
answer = rag.answer(
    "Summarise the structural-material (INOR-8) corrosion experience during "
    "the full MSRE operation: what was the average dissolved-Cr level in the "
    "fuel salt, did it change over the reactor lifetime, and how does it compare "
    "to the pre-MSRE laboratory predictions from corrosion capsule tests?"
)

answer = rag.answer(
    "Which fission products deposited preferentially on graphite vs. metal "
    "surfaces in the MSRE primary circuit? What operational conditions (power "
    "level, temperature, UF₃/UF₄ ratio) correlated with higher noble-metal "
    "deposition on graphite?"
)
```

---

## 7 — Supporting Future MSR Designs: Extracting Lessons Learned

**Paper connection:** The paper explicitly identifies lessons for future MSR
designs: the need for online redox monitoring, the importance of noble-gas
removal (Xe/Kr poisoning), the performance of graphite at high burnup, and the
suitability of INOR-8 for long-term service.

**Data-layer capability:** These lessons are stored as structured knowledge,
accessible to AI agents designing next-generation reactors.

```python
answer = rag.answer(
    "What were the key lessons learned from MSRE operation regarding: "
    "(1) salt-chemistry control for corrosion prevention, "
    "(2) noble-gas removal system design, "
    "(3) graphite moderator lifetime, and "
    "(4) pump and heat-exchanger reliability?"
)

answer = rag.answer(
    "What modifications to INOR-8 (Hastelloy N) were recommended after the MSRE "
    "based on the tellurium embrittlement observed, and what alloy development "
    "programme followed the MSRE shutdown?"
)
```

---

## Summary: Haubenreich & Engel (1970) × Data-Layer Capability

| Experimental / operational phase | Data ingested | Data-layer capability |
|---|---|---|
| Design-basis retrieval | ORNL pre-MSRE reports | `rag.load_msr_archive()` |
| Continuous process monitoring | Temps, power, flow, pump speed | `loader.ingest_sensor_snapshot()` |
| Online chemistry monitoring | UF₃/UF₄, dissolved Cr/Fe, off-gas | `loader.ingest_sensor_snapshot()` |
| Maintenance and fuel-addition events | Be additions, salt processing, repairs | `loader.ingest_text()` (event_log) |
| Post-campaign specimen analysis | Noble-metal/fission-product deposits | `loader.ingest_text()` (characterisation_report) |
| Cross-campaign synthesis | Full 13 172 h dataset | `rag.answer()` |
| Lessons-learned extraction | Design recommendations for future MSRs | `rag.answer()` over archive |
