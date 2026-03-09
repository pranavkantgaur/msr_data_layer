# How the MSR Data Layer Assists the McCoy et al. (1970) Tellurium Embrittlement Study

> **Paper:** McCoy H.E., Beatty R.L., Cook W.H., Gehlbach R.E., Kennedy C.R.,
> Koger J.W., Lof A.J., Weir J.R., Whitman G.D. — *"New Developments in
> Materials for Molten-Salt Reactors"*, *Nuclear Applications and Technology*,
> Vol. 8, No. 2, pp. 156–169, 1970.
>
> The landmark study that identified and characterised **tellurium-induced
> intergranular cracking** as the primary post-MSRE materials concern for
> Hastelloy N structural components.  The paper reports:
> * Post-MSRE examination of Hastelloy N coupons showing grain-boundary
>   cracking and tellurium penetration to >200 µm depth
> * Capsule irradiation tests relating crack severity to fission-product Te
>   concentration and salt oxidation potential (UF₃/UF₄ ratio)
> * Alloy modification programme: Ti, Nb, Hf additions to Hastelloy N as
>   mitigation strategies
> * Mechanical test results (bend tests, tensile tests, Charpy impact) on
>   embrittled vs. control specimens
> * Key mechanistic finding: **Te attack severity is suppressed at higher
>   UF₃/UF₄ ratios** (more reducing salt conditions)

---

## 1 — Design Phase: Retrieving MSRE Context for Te Behaviour

**Paper connection:** Understanding Te embrittlement requires knowing the
MSRE operating conditions under which it occurred — particularly the UF₃/UF₄
ratio history, the Te fission yield in ²³⁵U fuel, and the locations within the
primary circuit where Te deposits were most severe.

**Data-layer capability:** Query the ORNL archive for MSRE operational
chemistry data relevant to tellurium behaviour.

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()

answer = rag.answer(
    "What was the UF₃/UF₄ ratio history in the MSRE fuel salt? "
    "When did the ratio drop below 0.005 (oxidising conditions), and "
    "how did the team restore reducing conditions? "
    "What fission yield of Te-132 was expected at MSRE power levels?"
)
print(answer)

answer = rag.answer(
    "Where in the MSRE primary circuit was the heaviest tellurium deposition "
    "observed? Was it correlated with temperature, flow velocity, or material "
    "surface area? Cite the ORNL post-irradiation examination reports."
)
```

---

## 2 — During Capsule Tests: Logging Irradiation Conditions

**Paper connection:** McCoy's embrittlement capsule tests were conducted in
the HFIR (High Flux Isotope Reactor) and in the MSRE itself, with specimens
exposed to defined fast/thermal neutron fluxes, fission-product-doped salts,
and controlled UF₃/UF₄ ratios.  Accurate recording of irradiation conditions
is essential for interpretation of mechanical test results.

**Data-layer capability:** Capsule irradiation parameters are stored as sensor
snapshots, one per flux-monitoring interval.

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "1969-03-15T00:00Z",
     "sensor": "fast_neutron_flux_n_cm2_s", "value": 3.4e14,  "unit": "n/(cm²·s)",
     "capsule_id": "HASTELLOY-N-CAP-042"},
    {"timestamp": "1969-03-15T00:00Z",
     "sensor": "thermal_neutron_flux",       "value": 2.1e14,  "unit": "n/(cm²·s)",
     "capsule_id": "HASTELLOY-N-CAP-042"},
    {"timestamp": "1969-03-15T00:00Z",
     "sensor": "capsule_temp_c",             "value": 690.0,   "unit": "°C",
     "capsule_id": "HASTELLOY-N-CAP-042"},
    {"timestamp": "1969-03-15T00:00Z",
     "sensor": "uf3_uf4_ratio_target",       "value": 0.015,   "unit": "mol/mol",
     "capsule_id": "HASTELLOY-N-CAP-042"},
], source_id="irrad-CAP-042-1969-03-15")
```

---

## 3 — During Tests: Ingesting Te-Doping and Salt-Chemistry Records

**Paper connection:** McCoy's capsule tests used Te-doped salt to accelerate
embrittlement at controlled Te concentrations.  The UF₃/UF₄ ratio was set to
specific values (ranging from 0.001 to 0.10) to test the redox-potential
hypothesis.  These chemistry parameters must be linked to mechanical test
outcomes.

**Data-layer capability:** Salt preparation and Te-addition records are stored
as event logs tied to capsule IDs.

```python
loader.ingest_text(
    rag,
    text=(
        "Salt preparation — capsule CAP-042. "
        "Base salt: FLiBe (LiF 66 mol% – BeF₂ 34 mol%), 150 g. "
        "Tellurium addition: 50 ppm Te (as TeF4), to simulate MSRE fission-product "
        "inventory after ~5000 MWh operation. "
        "Redox adjustment: 0.8 g Be metal added to achieve UF₃/UF₄ = 0.015 "
        "(mildly reducing, representative of nominal MSRE chemistry). "
        "Post-preparation ICP check: Te 48 ppm, UF₃/UF₄ = 0.016 ± 0.002. "
        "Capsule sealed under Ar on 1969-03-10."
    ),
    source_id="salt-prep/CAP-042",
    data_type="salt_preparation_record",
)
```

---

## 4 — Post-Irradiation: Storing Mechanical Test Results

**Paper connection:** Post-irradiation examination of McCoy's specimens
included **bend tests** (measuring ductility loss as % reduction in ductility
relative to unirradiated control), **tensile tests** (yield strength, ultimate
tensile strength, elongation to fracture), and **Charpy impact tests** (energy
absorbed).  Results were tabulated vs. Te concentration and UF₃/UF₄ ratio.

**Data-layer capability:** Mechanical test results are stored as structured
characterisation records indexed by alloy, capsule, Te level, and redox
condition.

```python
# Bend-test result for standard Hastelloy N at high Te, low UF₃/UF₄
loader.ingest_text(
    rag,
    text=(
        "Bend test — Hastelloy N specimen HN-042-A (capsule CAP-042, "
        "standard alloy, 50 ppm Te, UF₃/UF₄ = 0.015, 700 °C, 2000 h MSRE irrad.). "
        "Bend angle to first crack: 22°. "
        "Unirradiated control bend angle: 180° (no cracking). "
        "Ductility retention: 12%. "
        "Cracking mode: intergranular. "
        "Te penetration depth (SEM): 180 µm."
    ),
    source_id="bend-test/CAP-042/HN-042-A",
    data_type="characterisation_report",
)

# Bend-test result for Ti-modified Hastelloy N (same conditions)
loader.ingest_text(
    rag,
    text=(
        "Bend test — Ti-modified Hastelloy N specimen HN-Ti-042-B (capsule CAP-042, "
        "Ti 0.5 wt% addition, 50 ppm Te, UF₃/UF₄ = 0.015, 700 °C, 2000 h). "
        "Bend angle to first crack: 95°. "
        "Ductility retention: 53%. "
        "Cracking mode: transgranular (intergranular crack density much reduced). "
        "Te penetration depth (SEM): 42 µm."
    ),
    source_id="bend-test/CAP-042/HN-Ti-042-B",
    data_type="characterisation_report",
)
```

---

## 5 — Cross-Test Analysis: Redox Potential vs. Embrittlement Severity

**Paper connection:** The key finding is the strong anti-correlation between
UF₃/UF₄ ratio and Te-crack depth: specimens at UF₃/UF₄ = 0.001 (highly
oxidising) showed >300 µm penetration and complete ductility loss, while
specimens at UF₃/UF₄ = 0.05 showed only ~20 µm penetration and retained >80%
ductility.

**Data-layer capability:** Once all capsule results are ingested, the RAG
pipeline can extract and plot the trend across the full parameter space.

```python
answer = rag.answer(
    "Summarise the relationship between UF₃/UF₄ ratio and Te-induced "
    "intergranular crack depth in Hastelloy N across all McCoy capsule tests. "
    "At what UF₃/UF₄ threshold does crack penetration fall below 50 µm? "
    "Does Ti modification shift this threshold?"
)

answer = rag.answer(
    "How does Te embrittlement severity (bend-test ductility retention) vary "
    "with Te concentration in the salt (10 ppm vs. 50 ppm vs. 200 ppm) at a "
    "fixed UF₃/UF₄ ratio of 0.015? Is the response linear or threshold-like?"
)
```

---

## 6 — Connecting to Online Redox Monitoring: Prevention Strategy

**Paper connection:** McCoy's mechanistic finding — that reducing conditions
suppress Te cracking — is the scientific basis for the MSRE/MSBR requirement
to maintain UF₃/UF₄ > 0.01 as a materials-protection operating limit.  This
limit must be linked to a real-time redox sensor alarm threshold.

**Data-layer capability:** The operational threshold discovered by McCoy is
encoded as an alarm limit in the data layer, linking experimental findings to
plant operations.

```python
# Define the materials-protection threshold as a monitoring event
loader.ingest_text(
    rag,
    text=(
        "Materials-protection operating limit (derived from McCoy et al. 1970). "
        "Parameter: fuel_salt_uf3_uf4_ratio. "
        "Lower alarm threshold: 0.010 (below this, Te crack penetration >100 µm "
        "expected within 2000 h at 700 °C based on McCoy capsule-test correlation). "
        "Action at alarm: immediate Be-metal addition to restore reducing conditions. "
        "Basis: McCoy, Nucl. Appl. Technol. 8(2), 1970, Fig. 7."
    ),
    source_id="operating-limit/uf3-uf4-te-embrittlement",
    data_type="operating_limit",
)

answer = rag.answer(
    "Based on McCoy et al. (1970), what is the recommended minimum UF₃/UF₄ ratio "
    "to maintain Hastelloy N ductility above 50% over a 20-year reactor lifetime? "
    "What online monitoring and control strategy should be implemented?"
)
```

---

## Summary: McCoy et al. (1970) × Data-Layer Capability

| Experimental phase | Data ingested | Data-layer capability |
|---|---|---|
| Design — MSRE Te/UF₃/UF₄ context | ORNL MSRE chemistry reports | `rag.load_msr_archive()` |
| During — irradiation conditions | Fast/thermal flux, temp, redox | `loader.ingest_sensor_snapshot()` |
| During — Te-doping records | Salt prep, Te level, UF₃/UF₄ set-point | `loader.ingest_text()` (salt_preparation_record) |
| Post-irradiation — mechanical tests | Bend, tensile, Charpy per specimen | `loader.ingest_text()` (characterisation_report) |
| Post-irradiation — SEM/EDS | Te penetration depth, crack morphology | `loader.ingest_text()` (characterisation_report) |
| Cross-test — redox vs. embrittlement | Full parameter-space dataset | `rag.answer()` |
| Operations — prevention strategy | UF₃/UF₄ alarm threshold, Be-addition protocol | `loader.ingest_text()` (operating_limit) |
