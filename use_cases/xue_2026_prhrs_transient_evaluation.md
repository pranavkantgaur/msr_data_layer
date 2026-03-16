# How the MSR Data Layer Supports the Xue et al. (2026) PRHRS Transient Evaluation Study

> **Paper:** Xue S., Zhou C., Yu H., Huang P., Zou Y. — *"Transient
> performance evaluation of passive residual heat removal system in liquid
> molten salt reactor"*, *Annals of Nuclear Energy*, 2026,
> DOI: https://doi.org/10.1016/j.anucene.2026.112227.
> Shanghai Institute of Applied Physics, Chinese Academy of Sciences /
> University of Chinese Academy of Sciences / State Key Laboratory of
> Thorium Energy, SINAP.
>
> The paper proposes a systematic set of **performance evaluation criteria**
> for the Secondary-Side Passive Residual Heat Removal System (SSPRHRS) of a
> **30 MWth liquid-fueled MSR (LF-MSR)** and validates those criteria against
> three design-basis accident scenarios using the **RELAP5-TMSR** code:
>
> * **Station blackout (SBO)** — all active power lost; primary and secondary
>   pumps trip; control rods insert; natural circulation is the sole cooling
>   mechanism.
> * **Primary pump seizure** — sudden loss of primary flow while secondary
>   circuit remains active.
> * **Secondary flow loss** — loss of secondary circuit flow while the primary
>   circuit continues operating.
>
> The reactor uses a three-loop architecture:
> * Primary loop: LiF-BeF₂-ZrF₄-UF₄-ThF₄ fuel salt (375 kg/s, 660–700 °C)
> * Secondary loop: NaF-BeF₂ coolant salt (115 kg/s)
> * Tertiary loop: helium Brayton cycle
>
> PRHRS performance criteria:
> * Fuel salt temperature ≤ 815 °C (material limit, UNS N10003 alloy)
> * Secondary salt temperature ≥ 425 °C (solidification prevention)
> * Passive heat removal capacity ≈ 2.5% of total reactor power (~750 kWth)
>
> Key findings:
> * Peak fuel salt temperature under SBO: 770.8 °C
> * Peak fuel salt temperature under pump seizure: 811.2 °C (near design limit)
> * Maximum measured PRHRS heat removal: 2.89 MW
> * All scenarios remained within safety limits through natural-circulation
>   cooling alone.

---

## 1 — Design Phase: Querying Historical Natural-Circulation and Decay-Heat Data

**Paper connection:** The paper's PRHRS design builds directly on the
natural-circulation cooling behaviour documented for the MSRE and MSBR.
Before running RELAP5-TMSR transient models, designers need baseline data:
measured natural-circulation driving heads in molten-salt loops, observed
thermal-hydraulic instabilities, and the correlation between fuel salt
temperature and heat rejection capacity.

**Data-layer capability:** Load the ORNL archive and query it for
natural-circulation and residual-heat-removal data.

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()   # one-time; re-runs add only new files

# Retrieve ORNL natural-circulation heat-removal data
answer = rag.answer(
    "What experimental data exist in the ORNL MSR programme on natural "
    "circulation heat removal in molten fluoride salt loops? Include "
    "measured flow velocities, temperature differentials, and any documented "
    "instabilities during pump-off or reduced-flow conditions."
)
print(answer)

# Retrieve decay-heat correlation data used in ORNL analyses
answer = rag.answer(
    "What decay-heat correlations and measured fission-product decay powers "
    "are documented in ORNL MSR reports for FLiBe or fluoride fuel salts? "
    "What fraction of full power was observed at 10 s, 100 s, and 1 hour "
    "after reactor shutdown in the MSRE or MSBR studies?"
)
print(answer)

# Retrieve ORNL data on secondary salt (NaF-BeF₂) thermal properties
answer = rag.answer(
    "What are the solidification temperatures, viscosities, and heat-transfer "
    "coefficients for NaF-BeF₂ coolant salt documented in ORNL MSR reports? "
    "What operating temperature margins above solidification were maintained "
    "in ORNL loop experiments to prevent salt freezing?"
)
print(answer)
```

```bash
python msr_digital_twin_with_rag.py \
  "Summarise ORNL data on passive or natural-circulation cooling in MSR loops"
```

---

## 2 — Simulation Validation: Ingesting RELAP5-TMSR Baseline Results

**Paper connection:** The paper's RELAP5-TMSR model establishes steady-state
initial conditions (reactor power 100%, fuel salt 660–700 °C, fuel flow
375 kg/s, secondary flow 115 kg/s) before launching each transient.
These baseline simulation outputs — temperatures, flow rates, pressures,
heat exchanger duties — form the validation reference that experimental
data and future runs must match.

**Data-layer capability:** Simulation results are ingested as structured
characterisation records with their source identified, enabling
cross-comparison with subsequent experimental measurements or updated
RELAP5-TMSR runs.

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Ingest RELAP5-TMSR steady-state baseline for LF-MSR
loader.ingest_text(
    rag,
    text=(
        "RELAP5-TMSR simulation result — LF-MSR steady-state baseline. "
        "Source: Xue et al., Ann. Nucl. Energy (2026), DOI 10.1016/j.anucene.2026.112227. "
        "Reactor thermal power: 30 MWth (100%). "
        "Fuel salt (LiF-BeF₂-ZrF₄-UF₄-ThF₄): "
        "  hot-leg temperature: 700 °C, cold-leg temperature: 660 °C, "
        "  mass flow rate: 375 kg/s. "
        "Secondary salt (NaF-BeF₂): "
        "  inlet temperature: 480 °C, outlet temperature: 540 °C, "
        "  mass flow rate: 115 kg/s. "
        "IHX duty: 30 MWth. "
        "PRHRS isolation valves: CLOSED (normal operation). "
        "Air damper: CLOSED (normal operation)."
    ),
    source_id="sim/relap5-tmsr/LF-MSR-2026/steady-state-baseline",
    data_type="characterisation_report",
)

# Ingest station blackout transient result
loader.ingest_text(
    rag,
    text=(
        "RELAP5-TMSR simulation result — Station Blackout (SBO) transient. "
        "Source: Xue et al., Ann. Nucl. Energy (2026), Table 3 / Figure 7. "
        "Initiating event: t=0 s, all power sources lost, primary pump trip, "
        "secondary pump trip, control-rod insertion, PRHRS activation. "
        "Peak fuel salt temperature: 770.8 °C at t≈60 s (< 815 °C material limit). "
        "Time to natural-circulation stabilisation: ~300 s. "
        "PRHRS maximum heat removal power: 2.88 MW. "
        "Secondary salt minimum temperature: 438 °C (> 425 °C freeze limit). "
        "Long-term fuel salt temperature at 3600 s: ~690 °C. "
        "Safety criteria satisfied: YES."
    ),
    source_id="sim/relap5-tmsr/LF-MSR-2026/SBO-transient",
    data_type="characterisation_report",
)

# Ingest main pump seizure transient result
loader.ingest_text(
    rag,
    text=(
        "RELAP5-TMSR simulation result — Primary pump seizure transient. "
        "Source: Xue et al., Ann. Nucl. Energy (2026), Table 3 / Figure 9. "
        "Initiating event: t=0 s, sudden primary pump seizure (flow → 0 in <1 s). "
        "Reactor scram on low-flow signal: t≈0.5 s. "
        "PRHRS activation: t≈1 s. "
        "Peak fuel salt temperature: 811.2 °C at t≈15 s (< 815 °C material limit). "
        "PRHRS maximum heat removal power: 2.89 MW. "
        "Secondary salt minimum temperature: 431 °C (> 425 °C freeze limit). "
        "Long-term stability: achieved at ~600 s. "
        "Safety criteria satisfied: YES (with <4 °C margin at peak)."
    ),
    source_id="sim/relap5-tmsr/LF-MSR-2026/pump-seizure-transient",
    data_type="characterisation_report",
)

# Ingest secondary flow loss transient result
loader.ingest_text(
    rag,
    text=(
        "RELAP5-TMSR simulation result — Secondary flow loss transient. "
        "Source: Xue et al., Ann. Nucl. Energy (2026), Table 3 / Figure 11. "
        "Initiating event: t=0 s, secondary pump trip (secondary flow → 0). "
        "Reactor scram on secondary-flow-low signal: t≈1 s. "
        "PRHRS activation: t≈2 s. "
        "Peak fuel salt temperature: below 815 °C. "
        "PRHRS maximum heat removal power: ~2.87 MW. "
        "Secondary salt minimum temperature: >425 °C. "
        "Safety criteria satisfied: YES."
    ),
    source_id="sim/relap5-tmsr/LF-MSR-2026/secondary-flow-loss-transient",
    data_type="characterisation_report",
)
```

---

## 3 — During Accident: Real-Time PRHRS Performance Monitoring

**Paper connection:** The three accident scenarios each produce a distinctive
time-series of temperatures, flow rates, and heat removal powers across the
primary loop, IHX, secondary loop, and air-cooling tower.  During an actual
PRHRS activation on the LF-MSR prototype, these readings must be logged in
real time so that operators can confirm the system is behaving within the
evaluated envelope and that the fuel salt temperature is not approaching the
815 °C material limit.

**Data-layer capability:** `PlantDataLoader.ingest_sensor_snapshot()` stores
each instrument scan alongside its reactor and accident state.

```python
# Log a PRHRS activation snapshot at t=30 s during an SBO event
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2026-09-01T03:00:30Z",
     "sensor": "fuel_salt_hot_leg_temp_c",     "value": 758.4,  "unit": "°C",
     "reactor_id": "LF-MSR-001",
     "accident_type": "station_blackout",
     "t_since_scram_s": 30},
    {"timestamp": "2026-09-01T03:00:30Z",
     "sensor": "fuel_salt_cold_leg_temp_c",    "value": 672.1,  "unit": "°C",
     "reactor_id": "LF-MSR-001",
     "accident_type": "station_blackout",
     "t_since_scram_s": 30},
    {"timestamp": "2026-09-01T03:00:30Z",
     "sensor": "prhrs_heat_removal_kw",        "value": 1843.0, "unit": "kW",
     "reactor_id": "LF-MSR-001",
     "accident_type": "station_blackout",
     "t_since_scram_s": 30},
    {"timestamp": "2026-09-01T03:00:30Z",
     "sensor": "secondary_salt_min_temp_c",    "value": 445.2,  "unit": "°C",
     "reactor_id": "LF-MSR-001",
     "accident_type": "station_blackout",
     "t_since_scram_s": 30},
    {"timestamp": "2026-09-01T03:00:30Z",
     "sensor": "natural_circ_fuel_flow_kg_s",  "value": 12.4,   "unit": "kg/s",
     "reactor_id": "LF-MSR-001",
     "accident_type": "station_blackout",
     "t_since_scram_s": 30},
    {"timestamp": "2026-09-01T03:00:30Z",
     "sensor": "air_cooling_tower_temp_c",     "value": 62.3,   "unit": "°C",
     "reactor_id": "LF-MSR-001",
     "accident_type": "station_blackout",
     "t_since_scram_s": 30},
], source_id="LF-MSR-001-SBO-20260901T030000Z-t30s")

# After full transient, query to confirm safety criteria were met
answer = rag.answer(
    "For the station blackout event on LF-MSR-001 on 2026-09-01, "
    "what was the maximum recorded fuel salt temperature, did it remain "
    "below the 815 °C material limit, and what was the minimum secondary "
    "salt temperature compared to the 425 °C freeze threshold?"
)
print(answer)
```

---

## 4 — Performance Criteria Verification: Automated Limit Checking

**Paper connection:** The paper defines three quantitative PRHRS performance
criteria that must be satisfied under all three accident scenarios:
(1) fuel salt temperature ≤ 815 °C, (2) secondary salt temperature ≥ 425 °C,
and (3) PRHRS heat removal capacity ≥ 2.5% of reactor power (~750 kWth).
These criteria must be verifiable from the operational data record both for
regulatory submissions and for post-accident reviews.

**Data-layer capability:** Criteria limits are ingested as structured
specification records; the RAG pipeline then answers automated compliance
queries against ingested sensor histories.

```python
# Store PRHRS performance criteria as a reference specification
loader.ingest_text(
    rag,
    text=(
        "PRHRS performance specification — LF-MSR 30 MWth. "
        "Source: Xue et al., Ann. Nucl. Energy (2026), Section 4 — "
        "Performance Evaluation Criteria. "
        "Criterion 1 (Adequate decay heat removal): "
        "  Maximum fuel salt temperature ≤ 815 °C (UNS N10003 structural limit). "
        "Criterion 2 (Anti-freezing): "
        "  Secondary salt (NaF-BeF₂) temperature ≥ 425 °C at all circuit locations. "
        "Criterion 3 (Heat removal capacity): "
        "  PRHRS must remove ≥ 2.5% of rated thermal power = 750 kWth under "
        "  accident conditions; demonstrated maximum 2.89 MW. "
        "Criterion 4 (System height): "
        "  PRHRS structural height ≤ 40 m (civil engineering constraint). "
        "Design-basis accidents evaluated: SBO, primary pump seizure, "
        "secondary flow loss."
    ),
    source_id="prhrs-spec/LF-MSR-001/performance-criteria-Xue2026",
    data_type="characterisation_report",
)

# Automated compliance query over stored operational data
answer = rag.answer(
    "Retrieve all ingested transient data for LF-MSR accidents. "
    "For each accident scenario, confirm whether: "
    "(1) peak fuel salt temperature remained ≤ 815 °C, "
    "(2) secondary salt temperature remained ≥ 425 °C, and "
    "(3) PRHRS heat removal reached ≥ 750 kWth. "
    "Flag any exceedance with the magnitude and timestamp."
)
print(answer)
```

---

## 5 — Accident Event Logging: PRHRS Activation and State Transitions

**Paper connection:** The paper describes the PRHRS activation sequence:
on loss of power the isolation valve opens, the air damper opens under
gravity, natural air circulation begins, and heat is transferred to the
atmosphere.  For a real reactor, this activation sequence, its timing,
and any deviations must be logged as a traceable event record.

**Data-layer capability:** Event logs capture the activation sequence
with timestamps and actor information so that the sequence can be reconstructed
for root-cause analysis and regulatory reporting (Licensee Event Reports).

```python
# Log PRHRS activation event
loader.ingest_text(
    rag,
    text=(
        "PRHRS activation event log — LF-MSR-001. "
        "Event ID: LF-MSR-001-EVENT-20260901-001. "
        "Type: Station blackout (design basis). "
        "t=0 s (2026-09-01T03:00:00Z): Grid trip detected; all AC power lost. "
        "t=0.1 s: Reactor protection system activates; control rods begin insertion. "
        "t=0.3 s: Primary pump coast-down begins. "
        "t=0.5 s: Reactor scram confirmed (power < 1% of nominal). "
        "t=0.8 s: PRHRS isolation valve opens (gravity-actuated). "
        "t=1.2 s: Air cooling-tower damper opens (gravity-actuated). "
        "t=5.0 s: Natural air circulation established (confirmed by air-flow sensor). "
        "t=60 s: Peak fuel salt temperature reached: 770.8 °C (< 815 °C limit). "
        "t=300 s: Natural circulation in fuel salt loop fully established. "
        "t=3600 s: Fuel salt temperature stable at ~690 °C. "
        "Outcome: All safety criteria satisfied. No operator action required. "
        "Shift supervisor: T. Chen. Logged by: automated data acquisition system."
    ),
    source_id="event-log/LF-MSR-001/SBO-20260901-activation-sequence",
    data_type="event_log",
)

# Query for regulatory reporting
answer = rag.answer(
    "For the station blackout event on LF-MSR-001 on 2026-09-01, "
    "reconstruct the PRHRS activation sequence with timestamps. "
    "Was the peak fuel salt temperature reached before or after natural "
    "circulation was established? What was the total time from loss of power "
    "to stable long-term cooling?"
)
print(answer)
```

---

## 6 — Simulation-Experiment Cross-Validation: Comparing RELAP5-TMSR Against Prototype Data

**Paper connection:** The RELAP5-TMSR code predictions (peak temperatures,
heat removal powers, natural-circulation flow rates) will ultimately need to
be validated against measurements from the LF-MSR prototype.  The data layer
is the integration point where simulation results and experimental data share
the same vector store, enabling systematic comparison.

**Data-layer capability:** Both simulation and experimental records are
queried together, with the source distinguishable by the `source_id` prefix
(`sim/` vs. `exp/`).

```python
# Cross-validation query: simulation vs. experimental observation
answer = rag.answer(
    "Compare the RELAP5-TMSR simulation prediction for peak fuel salt "
    "temperature during primary pump seizure (source: Xue et al. 2026) "
    "with any experimentally measured peak temperatures from prototype "
    "PRHRS activation tests ingested in the data layer. "
    "What is the difference in degrees Celsius, and which source predicts "
    "a higher peak? Is the remaining margin to the 815 °C material limit "
    "consistent between simulation and experiment?"
)
print(answer)

# Compare natural-circulation stabilisation time
answer = rag.answer(
    "What natural-circulation flow stabilisation time was predicted by "
    "RELAP5-TMSR for the station blackout scenario on the 30 MWth LF-MSR? "
    "If prototype commissioning data on natural-circulation behaviour in "
    "NaF-BeF₂ or FLiBe loops are available in the knowledge base, compare "
    "the predicted and measured stabilisation times and driving heads."
)
print(answer)

# Query ORNL MSRE data for historical comparison
answer = rag.answer(
    "Were there any pump-trip or loss-of-forced-flow tests conducted on the "
    "MSRE or any ORNL molten-salt loop that measured the natural-circulation "
    "flow rate and temperature rise in the fuel salt following pump shutdown? "
    "How does the measured peak temperature rise compare to the 811.2 °C "
    "peak predicted by RELAP5-TMSR for the LF-MSR pump seizure accident?"
)
print(answer)
```

---

## 7 — Long-Term Safety Monitoring: Decay-Heat Tracking After Shutdown

**Paper connection:** The paper demonstrates that the PRHRS must remove decay
heat for an extended period after reactor shutdown.  Decay heat follows a
sum-of-exponentials correlation and gradually decreases over hours and days.
Confirming that PRHRS heat removal capacity exceeds the remaining decay heat
at all times post-accident is a continuous monitoring requirement, not a
one-time check.

**Data-layer capability:** Periodic sensor snapshots log the evolving
relationship between PRHRS heat removal power and inferred decay heat,
enabling automated queries that flag any shortfall.

```python
# Log PRHRS and decay-heat state at 1 h, 6 h, 24 h post-accident
for t_hours, fuel_temp, prhrs_kw, decay_kw in [
    (1,  682.4, 1120.0, 890.0),
    (6,  665.3,  740.0, 580.0),
    (24, 649.8,  410.0, 310.0),
]:
    loader.ingest_sensor_snapshot(rag, [
        {"timestamp": f"2026-09-01T{3+t_hours:02d}:00:00Z",
         "sensor": "fuel_salt_hot_leg_temp_c",  "value": fuel_temp,
         "unit": "°C", "reactor_id": "LF-MSR-001",
         "t_since_scram_h": t_hours, "accident_type": "station_blackout"},
        {"timestamp": f"2026-09-01T{3+t_hours:02d}:00:00Z",
         "sensor": "prhrs_heat_removal_kw",     "value": prhrs_kw,
         "unit": "kW", "reactor_id": "LF-MSR-001",
         "t_since_scram_h": t_hours, "accident_type": "station_blackout"},
        {"timestamp": f"2026-09-01T{3+t_hours:02d}:00:00Z",
         "sensor": "inferred_decay_heat_kw",    "value": decay_kw,
         "unit": "kW", "reactor_id": "LF-MSR-001",
         "t_since_scram_h": t_hours, "accident_type": "station_blackout"},
    ], source_id=f"LF-MSR-001-SBO-20260901-t{t_hours}h")

# Automated safety check: PRHRS capacity must always exceed decay heat
answer = rag.answer(
    "For the station blackout event on LF-MSR-001 on 2026-09-01, "
    "at each logged time point (1 h, 6 h, 24 h post-scram), did the PRHRS "
    "heat removal power exceed the inferred decay heat? What was the margin "
    "in kW at each point, and is the system trending toward a safe long-term "
    "steady state?"
)
print(answer)
```

---

## Summary: Xue et al. (2026) × Data-Layer Capability

| Experimental / operational phase | Data ingested | Data-layer capability |
|---|---|---|
| Pre-design ORNL data retrieval | Natural-circulation, decay-heat, NaF-BeF₂ data | `rag.answer()` over ORNL archive |
| Simulation baseline storage | RELAP5-TMSR steady-state and transient results | `loader.ingest_text()` (characterisation_report) |
| Real-time accident monitoring | Per-second temperatures, flows, PRHRS power | `loader.ingest_sensor_snapshot()` |
| Automated criteria verification | PRHRS limits (815 °C, 425 °C, 750 kWth) | `rag.answer()` compliance query |
| PRHRS activation event log | Activation sequence with timestamps | `loader.ingest_text()` (event_log) |
| Simulation–experiment cross-validation | RELAP5-TMSR vs. prototype measurements | `rag.answer()` spanning both source types |
| Long-term decay-heat tracking | Hourly PRHRS power vs. inferred decay heat | `loader.ingest_sensor_snapshot()` + `rag.answer()` |
