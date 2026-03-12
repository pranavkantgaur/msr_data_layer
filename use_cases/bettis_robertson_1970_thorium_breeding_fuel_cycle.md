# How the MSR Data Layer Accelerates Copenhagen Atomics' Thorium Fuel Cycle R&D

> **Paper:** Bettis E.S. & Robertson R.C. — *"The Design and Performance
> Features of a Single-Fluid Molten-Salt Breeder Reactor"*, *Nuclear
> Applications & Technology*, Vol. 8, pp. 190–207, 1970.
>
> The canonical ORNL design study for a single-fluid Molten Salt Breeder
> Reactor (MSBR) operating on the **thorium–uranium-233 fuel cycle**.  It
> establishes the quantitative relationships between fuel-salt composition
> (LiF-BeF₂-ThF₄-UF₄), online fuel processing, fission-product removal, and
> the achievable **breeding ratio** — the ratio of new fissile U-233 produced
> to fissile material consumed.
>
> Key technical contributions:
> * Thorium breeding chain: Th-232 + n → Th-233 (22 min) → Pa-233 (27 d) → U-233
> * **Pa-233 management strategy**: side-stream removal to a decay tank to
>   prevent neutron absorption before Pa-233 decays to U-233
> * Online processing schedule: noble-gas stripping (Kr/Xe, 30 s cycle),
>   noble-metal deposition on graphite, Pa removal (10 d cycle), rare-earth
>   fluoride removal (50 d cycle)
> * Breeding ratio sensitivity to Pa-233 inventory, Pa removal rate, and
>   fission-product neutron poison fraction
> * Fuel-salt inventory and ThF₄ replenishment economics
>
> **Why this matters to Copenhagen Atomics:** Copenhagen Atomics' WATT reactor
> uses **LiThF (LiF-ThF₄)** fuel salt with continuous online processing — the
> same paradigm Bettis & Robertson quantified for the MSBR.  Every breeding-
> ratio decision, Pa-233 removal schedule, and salt-chemistry target in the
> WATT design depends on the thermodynamic and neutronic data that Bettis &
> Robertson compiled.  The data layer connects Copenhagen Atomics' own
> experimental measurements (including their 2025 LiThF corrosion programme)
> to this ORNL baseline in a single queryable knowledge base.

---

## 1 — Design Phase: Querying the ORNL Thorium Breeding-Cycle Data

**Paper connection:** Before specifying their Pa-233 removal schedule or
ThF₄/UF₄ target ratios, Copenhagen Atomics' engineers must know what Bettis &
Robertson computed for breeding gain as a function of Pa removal rate constant,
and how sensitive that gain is to neutron poison fraction from rare-earth
fission products.

**Data-layer capability:** Load the ORNL OCR archive and query it for
breeding-cycle parameters directly.

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()   # one-time; re-runs add only new files

answer = rag.answer(
    "In the Bettis & Robertson MSBR design study, what breeding ratio was "
    "achieved as a function of the protactinium side-stream removal rate? "
    "What was the Pa inventory in the primary salt at equilibrium, and how "
    "long did Pa-233 need to remain in the decay tank before returning U-233 "
    "to the primary circuit? Include ORNL report numbers."
)
print(answer)

answer = rag.answer(
    "What fraction of fissile neutrons were absorbed by fission-product "
    "poisons (rare earths, Xe-135, Sm-149) in the MSBR design? How quickly "
    "did noble-gas stripping remove Kr and Xe from the fuel salt, and what "
    "was the assumed bubble-contactor efficiency? Cite ORNL-4541 or ORNL-4812."
)
print(answer)
```

```bash
# Equivalent CLI query
python msr_digital_twin_with_rag.py \
  "Summarise the MSBR online processing schedule: Pa removal, noble-gas \
   stripping, rare-earth removal cycle times and their impact on breeding ratio"
```

This retrieves quantitative MSBR benchmarks in seconds — replacing days of
manual searching through ORNL-4541, ORNL-4812, and related reports — and
gives Copenhagen Atomics a traceable design-basis for their WATT reactor
processing schedule.

---

## 2 — Design Phase: Deriving Fuel-Salt Composition Targets for LiThF

**Paper connection:** The MSBR used LiF-BeF₂-ThF₄-UF₄.  Copenhagen Atomics'
LiThF (LiF-ThF₄) avoids beryllium for safety and regulatory reasons.  Before
running new salt-property measurements, engineers need to know how the Bettis &
Robertson thermodynamic and neutronic optimisation results translate from
FLiBe-based salt to a BeF₂-free formulation.

**Data-layer capability:** Once academic literature is updated via OpenAlex,
the RAG pipeline spans both ORNL heritage data and recent LiThF studies.

```python
import subprocess
# Refresh literature index (pulls recent LiThF / thorium-MSR papers)
subprocess.run(["python", "msr_kb_sources.py", "--update-openalex"])

answer = rag.answer(
    "How does removing BeF₂ from the MSBR fuel-salt formulation affect: "
    "(a) the critical fissile loading (mol% UF₄) for a given ThF₄/LiF ratio, "
    "(b) the melting point and viscosity of LiF-ThF₄ compared to LiF-BeF₂-ThF₄, "
    "and (c) the neutron moderation contribution from the salt? "
    "Reference both ORNL data and post-2010 LiThF literature."
)
print(answer)

answer = rag.answer(
    "What ThF₄ mole fraction in LiF-ThF₄ (LiThF) minimises corrosion of "
    "316L stainless steel while maintaining sufficient thermal conductivity "
    "and a liquidus below 550 °C? Cite Cantor 1968 physical-properties data "
    "and the Lucas et al. 2025 LiThF corrosion results."
)
print(answer)
```

---

## 3 — During Operation: Logging Continuous Neutron Flux and Breeding Monitors

**Paper connection:** Bettis & Robertson showed that the achievable breeding
ratio depends critically on the real-time neutron economy — the fraction of
neutrons captured by Pa-233 before it decays.  In an operating WATT reactor,
continuous in-core neutron detectors and activation monitors track the neutron
flux profile; this data is compared against the design breeding model.

**Data-layer capability:** Periodic neutron-monitor readings are ingested as
sensor snapshots linked to the operating cycle identifier.

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Log thermal neutron flux, power, and Pa-233 activation monitor every hour
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-01-15T10:00Z",
     "sensor": "core_avg_thermal_flux_n_cm2_s", "value": 3.2e14, "unit": "n/(cm²·s)",
     "cycle_id": "WATT-CYCLE-001"},
    {"timestamp": "2026-01-15T10:00Z",
     "sensor": "reactor_power_MW_th",            "value": 22.4,   "unit": "MW(th)",
     "cycle_id": "WATT-CYCLE-001"},
    {"timestamp": "2026-01-15T10:00Z",
     "sensor": "pa233_activation_monitor_cps",   "value": 4870.0, "unit": "counts/s",
     "instrument": "in-salt-gamma-detector",
     "cycle_id": "WATT-CYCLE-001"},
    {"timestamp": "2026-01-15T10:00Z",
     "sensor": "th232_concentration_mol_pct",    "value": 11.8,   "unit": "mol%",
     "salt_system": "LiThF-primary",
     "cycle_id": "WATT-CYCLE-001"},
    {"timestamp": "2026-01-15T10:00Z",
     "sensor": "u233_concentration_mol_pct",     "value": 0.34,   "unit": "mol%",
     "salt_system": "LiThF-primary",
     "cycle_id": "WATT-CYCLE-001"},
], source_id="WATT-CYCLE-001-2026-01-15T10Z")
```

Once several months of readings are ingested, a query such as *"Plot the
U-233 mole fraction trend over WATT cycle 001 — is it increasing (net
breeding), decreasing (net burning), or flat?"* is directly answerable from
the stored time-series.

---

## 4 — During Operation: Recording Pa-233 Removal Side-Stream Events

**Paper connection:** The MSBR process removes Pa-233 on a ~10-day cycle by
diverting a fraction of the fuel salt to a fluoride-volatility unit, stripping
UF₄ and UF₅ as volatile UF₆, and returning the Pa-bearing residue to a
separate decay tank.  After ~7 half-lives (~200 days) the Pa-233 has fully
decayed to U-233 and is returned to the primary circuit.  Each removal and
return event is a critical breeding-cycle milestone that must be logged for
neutron-economy accounting.

**Data-layer capability:** Each Pa side-stream event is recorded as an
event log with mass-balance data, enabling audit and breeding-ratio calculation.

```python
# Log a Pa-233 side-stream withdrawal event
loader.ingest_text(
    rag,
    text=(
        "Pa-233 side-stream event — 2026-02-01T06:00Z. "
        "Withdrawal: 12.4 L of primary LiThF fuel salt diverted to Pa decay tank PT-001. "
        "Pa-233 estimated inventory at withdrawal: 8.3 mg (from activation monitor calibration). "
        "UF₄ stripped by fluoride volatility: 0.42 g UF₄ → 0.36 g UF₆ collected. "
        "Pa-bearing salt returned to decay tank for 27-day hold. "
        "Primary salt ThF₄ concentration post-withdrawal: 11.6 mol%. "
        "Operator: E. Andersen. Approved by: Fuel Chemistry Supervisor."
    ),
    source_id="event/pa-sidestream/WATT-CYCLE-001/20260201",
    data_type="event_log",
)

# Log the U-233 return event (~90 days later, after Pa decay is complete)
loader.ingest_text(
    rag,
    text=(
        "Pa decay-tank return event — 2026-05-02T08:00Z. "
        "Decay tank PT-001 salt returned to primary circuit after 90-day hold. "
        "U-233 recovered: 8.1 mg (98% Pa → U-233 conversion assumed). "
        "Pa-233 residual at return: <0.1 mg (confirmed by gamma spectroscopy). "
        "Primary salt U-233 concentration post-return: 0.38 mol% (up from 0.34 mol%). "
        "Operator: E. Andersen."
    ),
    source_id="event/pa-decay-return/WATT-CYCLE-001/20260502",
    data_type="event_log",
)
```

---

## 5 — During Operation: Logging Noble-Gas (Kr/Xe) Stripping Data

**Paper connection:** Xe-135 is the dominant short-term fission-product poison
(σ_a = 2.65 × 10⁶ barn).  The MSBR design assumed a 30-second bubble
contactor cycle to remove dissolved Kr and Xe before they could significantly
absorb neutrons.  Stripping efficiency determines how much of the theoretical
breeding gain is actually achieved.

**Data-layer capability:** Continuous off-gas monitor readings are ingested as
sensor snapshots, providing real-time poison inventory for breeding-ratio
tracking.

```python
# Log noble-gas off-gas monitor every 5 minutes
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-01-15T10:05Z",
     "sensor": "xe135_off_gas_activity_Bq_s",   "value": 2.4e10, "unit": "Bq/s",
     "instrument": "off-gas-gamma-monitor",
     "cycle_id": "WATT-CYCLE-001"},
    {"timestamp": "2026-01-15T10:05Z",
     "sensor": "kr85_off_gas_activity_Bq_s",     "value": 8.7e8,  "unit": "Bq/s",
     "instrument": "off-gas-gamma-monitor",
     "cycle_id": "WATT-CYCLE-001"},
    {"timestamp": "2026-01-15T10:05Z",
     "sensor": "helium_sparge_flow_slpm",        "value": 4.2,    "unit": "sl/min",
     "instrument": "mass-flow-controller-HE01",
     "cycle_id": "WATT-CYCLE-001"},
    {"timestamp": "2026-01-15T10:05Z",
     "sensor": "bubble_contactor_efficiency_pct", "value": 94.1,  "unit": "%",
     "derived_from": "xe135_off_gas_activity_Bq_s",
     "cycle_id": "WATT-CYCLE-001"},
], source_id="WATT-CYCLE-001-offgas-2026-01-15T10-05Z")
```

A query like *"What was the mean Xe-135 stripping efficiency over the first
500 h of WATT cycle 001, and were there any periods where the bubble contactor
efficiency dropped below 90%?"* is directly answerable from these records.

---

## 6 — Post-Campaign: Calculating Measured Breeding Ratio vs. MSBR Design Basis

**Paper connection:** Bettis & Robertson predicted a breeding ratio of 1.06 for
the reference MSBR at equilibrium.  For Copenhagen Atomics to validate their
WATT reactor design, they must calculate the *measured* breeding ratio from
their actual Pa-removal and U-233 return records, and compare it against both
the Bettis & Robertson model and any reduction attributable to their 316L SS
structural material (which has higher parasitic neutron absorption than
Hastelloy N's nickel base).

**Data-layer capability:** Once sensor history (neutron flux, U-233/Th-232
concentrations) and processing event logs (Pa removal/return, fission-product
stripping) are all ingested, the RAG pipeline synthesises a measured breeding
summary.

```python
# Ingest end-of-cycle fuel-salt analysis
loader.ingest_text(
    rag,
    text=(
        "End-of-cycle-001 fuel salt analysis — 2026-06-01. "
        "Primary LiThF salt composition after 500 effective full-power hours: "
        "ThF₄: 11.4 mol% (initial: 11.8 mol%), UF₄: 0.41 mol% (initial: 0.34 mol%). "
        "Estimated U-233 produced from Th-232 breeding during cycle: 0.82 kg. "
        "Estimated U-233 consumed by fission: 0.61 kg. "
        "Net fissile gain: +0.21 kg → gross breeding ratio = 1.34 (cycle average). "
        "Fission-product neutron poison fraction: 1.8% (from Xe/Kr stripping records). "
        "Cycle duration: 500 h effective full power. Lab: Copenhagen Atomics Fuel Chemistry."
    ),
    source_id="cycle-analysis/WATT-CYCLE-001/EOC",
    data_type="characterisation_report",
)

answer = rag.answer(
    "Based on the end-of-cycle-001 analysis for the WATT reactor, what was "
    "the measured gross breeding ratio? How does it compare to the Bettis & "
    "Robertson MSBR prediction of 1.06? What explains the difference — "
    "different ThF₄ mole fraction, 316L vs Hastelloy N parasitic absorption, "
    "Pa-233 removal efficiency, or noble-gas stripping performance?"
)
print(answer)

answer = rag.answer(
    "From the WATT cycle-001 Pa-removal event logs and neutron flux records, "
    "what fraction of Pa-233 produced was captured in the decay tank before "
    "absorbing a neutron? How does this compare to the 95% capture efficiency "
    "assumed in the MSBR design basis?"
)
print(answer)
```

---

## 7 — Cross-Reference: Connecting Copenhagen Atomics' LiThF Data with ORNL FLiBe Baseline

**Paper connection:** The MSBR design was optimised for LiF-BeF₂-ThF₄-UF₄.
Copenhagen Atomics' choice of BeF₂-free LiThF means that every thermodynamic,
transport, and neutronic parameter must be cross-referenced against the ORNL
FLiBe baseline.  The data layer holds both the Bettis & Robertson MSBR data
and Copenhagen Atomics' own 2025 LiThF corrosion results in the same knowledge
base, enabling direct comparisons.

**Data-layer capability:** Combined query across ORNL archive (Bettis 1970,
Cantor 1968) and the ingested Copenhagen Atomics experimental data.

```python
answer = rag.answer(
    "Summarise the key physical property differences between the MSBR baseline "
    "salt (LiF-BeF₂-ThF₄-UF₄, 72-16-12-0.3 mol%) and Copenhagen Atomics' "
    "LiThF (LiF-ThF₄) regarding: liquidus temperature, viscosity at 700 °C, "
    "thermal conductivity, and uranium solubility limit. "
    "Use Cantor (1968) for the FLiBe baseline and recent LiThF literature for "
    "the comparison."
)
print(answer)

answer = rag.answer(
    "The Lucas et al. 2025 paper measured 316L SS corrosion in LiThF at 700 °C. "
    "The Koger 1972 loop tests measured Hastelloy N corrosion in FLiBe at 700 °C. "
    "Which material-salt combination gives lower dissolved Cr, Fe, and Ni "
    "concentrations at 3 000 h? What salt-chemistry factor (redox potential, "
    "UF₄ oxidant activity, ThF₄ vs BeF₂ effect) is responsible for the difference?"
)
print(answer)
```

---

## 8 — Regulatory Support: Generating Traceable Fuel-Cycle History Reports

**Paper connection:** Nuclear licensing requires comprehensive, auditable records
of fuel composition, processing operations, and breeding performance over the
entire operating history of the reactor.  The MSBR programme at ORNL generated
thousands of operational logs; a modern Copenhagen Atomics deployment must do
the same, with full traceability from raw sensor data through to breeding-ratio
conclusions.

**Data-layer capability:** All sensor snapshots, event logs, and
characterisation reports are stored with `source_id` metadata enabling
end-to-end audit queries.

```python
answer = rag.answer(
    "Generate a chronological summary of all Pa-233 side-stream removal and "
    "return events for WATT cycle 001, including estimated U-233 mass recovered "
    "in each return, and calculate the cumulative U-233 breeding contribution "
    "from Pa removal over the cycle."
)
print(answer)

answer = rag.answer(
    "Were there any periods during WATT cycle 001 where the ThF₄ mole fraction "
    "fell below 11 mol% (the design minimum for adequate thorium inventory)? "
    "If so, when, for how long, and what corrective action was taken?"
)
print(answer)
```

---

## Summary: Bettis & Robertson (1970) × Copenhagen Atomics WATT Reactor × Data-Layer Capability

| Operational phase | Data ingested | Data-layer capability |
|---|---|---|
| Design — breeding-ratio & Pa management | ORNL MSBR reports (ORNL-4541, ORNL-4812) | `rag.load_msr_archive()` + `rag.answer()` |
| Design — LiThF vs FLiBe salt properties | Cantor 1968, OpenAlex LiThF papers | `rag.answer()` over ORNL + OpenAlex |
| During — neutron flux & power monitors | Hourly flux, power, Pa-233 activation | `loader.ingest_sensor_snapshot()` |
| During — U-233/Th-232 salt concentrations | Periodic salt samples, mol% tracking | `loader.ingest_sensor_snapshot()` |
| During — Pa-233 side-stream events | Withdrawal volume, Pa mass, U-233 return | `loader.ingest_text()` (event_log) |
| During — Kr/Xe noble-gas stripping | Off-gas gamma monitor, bubble contactor | `loader.ingest_sensor_snapshot()` |
| Post-campaign — measured breeding ratio | EOC salt analysis, cycle mass balance | `loader.ingest_text()` (characterisation_report) + `rag.answer()` |
| Regulatory — fuel cycle audit trail | All sources via `source_id` | `rag.answer()` with full provenance |
