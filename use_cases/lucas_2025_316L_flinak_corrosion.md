# How the MSR Data Layer Assists the Lucas et al. (2025) Corrosion Study

> **Paper:** Lucas N., Woods R., Crombleholme S., Vandanapu H., Beer C.,
> Sobel J., Steenberg T., Patel M.K. — *"Effect of Salt Purity on the
> Corrosion of 316L SS: Long-Term Studies in Molten FLiNaK and ThF₄–LiF"*,
> *Journal of Nuclear Materials* (2025), PII S0022311525007913.
> Copenhagen Atomics A/S & University of Liverpool.
>
> The paper reports 18 static-immersion corrosion tests of **316L stainless
> steel** coupons (Cr 16.9 wt%, Ni 10.7 wt%, Mo 2.6 wt%) in two molten
> fluoride salt systems — **FLiNaK at 600 °C** and **LiThF (ThF₄-LiF) at
> 700 °C** — comparing purified versus untreated salt over **1 000, 2 000,
> and 3 000 h**.  Analysis methods include ICP-OES (Cr/Fe/Ni in salt),
> mass change, SEM/EDS cross-sections, and GIXRD phase identification.

The sections below show exactly where the data layer plugs into each phase of
this experimental programme.

---

## 1 — Design Phase: Retrieving ORNL Baselines for 316L SS and INOR-8

**Paper connection:** The Introduction positions Copenhagen Atomics' 316L
work against the MSRE/MSBR heritage of **Inconel** and **INOR-8** data —
roughly 1 mm corrosion per 20 000 h.  Before running expensive 3 000 h tests
the researchers must know what the ORNL reports actually measured, and under
which salt-chemistry conditions those rates were achieved.

**Data-layer capability:** Ingest the ORNL OCR archive and query it directly.

```python
from msr_digital_twin_with_rag import MSRDigitalTwinRAG

rag = MSRDigitalTwinRAG()
rag.load_msr_archive()   # one-time; re-runs add only new files

answer = rag.answer(
    "What mass-loss or corrosion-depth data exist in the ORNL reports for "
    "316 stainless steel or INOR-8 coupons in FLiNaK at 600–700 °C? "
    "Include experiment duration, temperature, and salt purity conditions."
)
print(answer)
```

```bash
python msr_digital_twin_with_rag.py \
  "Summarise ORNL MSRE container-material corrosion data for austenitic steels"
```

This surfaces specific ORNL report numbers and measured values — avoiding
manual archive searches across dozens of 1960s technical reports — and
establishes a traceable baseline against which the new 316L data can be
benchmarked.

---

## 2 — Design Phase: Surveying Recent 316L / FLiNaK Literature

**Paper connection:** The paper cites competing work on chromium depletion in
FLiNaK and on the role of moisture-derived HF in stainless steel attack.  The
researchers need to know what corrosion depths and ICP-OES concentrations have
already been reported for 316L in fluoride salts so they can frame their
contribution and choose appropriate exposure durations.

**Data-layer capability:** The OpenAlex loader ingests papers matching MSR
corrosion queries into the same vector store as the ORNL archive.

```bash
python msr_kb_sources.py --update-openalex
```

```python
answer = rag.answer(
    "What chromium and iron dissolution rates have been reported for 316L SS "
    "in FLiNaK or FLiBe in the last 10 years? "
    "Include temperature, exposure time, and whether the salt was purified."
)
```

A single query now spans six decades of literature — ORNL reports from the
1960s and peer-reviewed papers from the 2010s–2020s — in one step.

---

## 3 — During Experiment: Logging Furnace Conditions for All 18 Tests

**Paper connection:** The 18 immersion tests (9 purified FLiNaK, 9 untreated
FLiNaK) were run at **600 °C** under **0.3 bar Ar overpressure** inside an
argon glovebox (<10 ppm O₂ and H₂O).  Long-duration experiments (up to
3 000 h ≈ 125 days) accumulate furnace controller logs, thermocouple readings,
and glovebox atmosphere readings that must be preserved alongside the coupon
results.

**Data-layer capability:** `PlantDataLoader.ingest_sensor_snapshot()` stores
periodic furnace-condition records in the RAG knowledge base so they can be
co-queried with characterisation data.

```python
from msr_kb_sources import PlantDataLoader

loader = PlantDataLoader()

# Call from the DAQ script every 4 h for the duration of each test
loader.ingest_sensor_snapshot(rag, [
    {"timestamp": "2024-06-01T04:00Z",
     "sensor": "furnace_temperature_c",   "value": 600.2, "unit": "°C",
     "test_id": "FLiNaK-purified-3000h"},
    {"timestamp": "2024-06-01T04:00Z",
     "sensor": "ar_overpressure_bar",     "value": 0.302, "unit": "bar",
     "test_id": "FLiNaK-purified-3000h"},
    {"timestamp": "2024-06-01T04:00Z",
     "sensor": "glovebox_o2_ppm",         "value": 7.1,   "unit": "ppm",
     "test_id": "FLiNaK-purified-3000h"},
], source_id="FLiNaK-purified-3000h-2024-06-01T04Z")
```

After 3 000 h a query like *"Were there any periods during the purified-FLiNaK
3 000 h run where the glovebox O₂ exceeded 10 ppm, and what was the furnace
temperature at those times?"* can be answered without manually scanning
controller log files.

---

## 4 — During Experiment: Ingesting Salt-Preparation and Purification Records

**Paper connection:** The key experimental variable is salt purity.  The
Copenhagen Atomics purification method (high-temperature treatment under inert
gas, resulting in moisture below detection and oxides <10 ppm) is what
separates the two coupon populations.  Documenting the purification batch
record for every test tube is essential for traceability and for future
statistical analysis of purity vs. corrosion depth.

**Data-layer capability:** Free-text purification and preparation records are
ingested as event logs so they are retrievable alongside the coupon results.

```python
loader.ingest_text(
    rag,
    text=(
        "Salt batch CA-FLiNaK-P-007 (purified). "
        "Composition: LiF 46.5 mol%, NaF 11.5 mol%, KF 42 mol%. "
        "Purification: 24 h at 500 °C under Ar flow, followed by HF/H₂ sparging. "
        "Post-purification assay: moisture below detection limit (<1 ppm), "
        "oxide impurities 6 ppm (ICP-OES). "
        "Loaded into test tube T-P-07 on 2024-05-15; "
        "four 316L coupons (IDs: P07-A, P07-B, P07-C, P07-D) suspended. "
        "Test start: 2024-05-15T10:00Z. Target exposure: 3000 h at 600 °C."
    ),
    source_id="salt-batch-CA-FLiNaK-P-007",
    data_type="salt_preparation_record",
)
```

This links a coupon ID to a specific salt batch, enabling a future query like
*"Which coupons came from salt batches with oxide impurities above 8 ppm, and
what were their corrosion depths?"*

---

## 5 — After Each Time-Point: Storing ICP-OES Results

**Paper connection:** Post-test salt samples were dissolved in nitric/
hydrochloric acid and analysed by ICP-OES for **Cr, Fe, and Ni**.  The paper
reports average concentrations across the three exposure durations (Table in
Section 4.3):

| Metal | Untreated (mg/kg) | Purified (mg/kg) |
|---|---|---|
| Cr | 1 200 ± 100 | 110 ± 6 |
| Fe | 800 ± 300  | 22 ± 7  |
| Ni | 80 ± 32    | below detection |

**Data-layer capability:** Each ICP-OES result set is ingested as a structured
characterisation record so it can be queried alongside corrosion-depth and
mass-change data.

```python
loader.ingest_text(
    rag,
    text=(
        "ICP-OES salt analysis — test tube T-U-03 (untreated FLiNaK, 1000 h, 600 °C). "
        "Dissolved metals: Cr 1185 mg/kg, Fe 510 mg/kg, Ni 48 mg/kg. "
        "Analysis date: 2024-07-20. Lab: Copenhagen Atomics internal. "
        "Coupons: U03-A, U03-B, U03-C, U03-D."
    ),
    source_id="icp-oes/T-U-03/1000h",
    data_type="characterisation_report",
)
```

After all 18 tests are ingested, a single RAG query surfaces the full Cr/Fe/Ni
dissolution trend across the purified vs. untreated series and all three
time-points — the same analysis the paper presents in Section 4.3, but
queryable in plain language.

---

## 6 — After Each Time-Point: Storing Mass-Change and Corrosion-Depth Records

**Paper connection:** The paper reports coupon mass change (untreated salt
~194× greater loss than purified) and SEM-measured corrosion depths (untreated
68.5 → 112.1 µm vs. purified 2.1 → 3.0 µm over 1 000–3 000 h), with
ImageJ used for depth measurements on cross-sectional SEM images.

**Data-layer capability:** Mass and depth records are ingested per coupon per
time-point with a consistent `source_id` scheme enabling cross-series
comparisons.

```python
# Mass-change record after the 1000 h untreated-FLiNaK test teardown
loader.ingest_text(
    rag,
    text=(
        "Mass change — coupon U03-A (untreated FLiNaK, 1000 h, 600 °C). "
        "Pre-exposure mass: 14.823 g. Post-exposure mass: 14.695 g. "
        "Mass loss: 128 mg. Coupon area: 13.24 cm². "
        "Specific mass loss: 9.67 mg/cm²."
    ),
    source_id="mass-change/T-U-03/U03-A/1000h",
    data_type="characterisation_report",
)

# SEM corrosion depth from ImageJ measurement
loader.ingest_text(
    rag,
    text=(
        "SEM cross-section — coupon U03-A (untreated FLiNaK, 1000 h, 600 °C). "
        "Intergranular corrosion observed. Maximum corrosion depth (ImageJ): 71 µm. "
        "Mean corrosion depth: 68.5 µm. "
        "Attack mode: intergranular; no uniform dissolution. "
        "Cr-depleted zone confirmed by EDS line scan."
    ),
    source_id="sem/T-U-03/U03-A/1000h",
    data_type="characterisation_report",
)
```

---

## 7 — Post-Exposure: Storing GIXRD Phase-Identification Results

**Paper connection:** GIXRD identified key phases that explain the mechanistic
difference between purified and untreated salts:

* **Purified FLiNaK coupons:** Cr₇C₃ and Cr₂₃C₆ (chromium carbides) —
  hypothesised to act as diffusion barriers.
* **Untreated FLiNaK coupons:** FeCr₂O₄ spinel, KF/K-Cr-F compounds, and
  γ-Fe → α-Fe transformation (austenite to ferrite) from Cr and Ni depletion.

**Data-layer capability:** Phase-identification results are ingested as
structured text, linking phase names to the coupon, salt condition, and
exposure duration, so that future queries can reason over mechanism.

```python
loader.ingest_text(
    rag,
    text=(
        "GIXRD phase identification — coupon P07-B (purified FLiNaK, 3000 h, 600 °C). "
        "Phases detected: Cr₇C₃ (chromium carbide), Cr₂₃C₆ (chromium carbide), "
        "γ-Fe (austenite, matrix retained). "
        "No FeCr₂O₄ detected. No KF or K-Cr-F compounds. "
        "Interpretation: Cr carbide surface film may act as diffusion barrier "
        "limiting further Cr dissolution into the salt."
    ),
    source_id="gixrd/T-P-07/P07-B/3000h",
    data_type="characterisation_report",
)

loader.ingest_text(
    rag,
    text=(
        "GIXRD phase identification — coupon U03-C (untreated FLiNaK, 3000 h, 600 °C). "
        "Phases detected: FeCr₂O₄ (spinel), KF, K-Cr-F compounds, α-Fe (ferrite). "
        "Original γ-Fe austenite peak greatly reduced — consistent with "
        "Cr and Ni depletion driving γ-to-α transformation. "
        "Interpretation: impurity-driven oxide dissolution removes passive Cr₂O₃ "
        "layer, exposing alloy to further fluoride attack."
    ),
    source_id="gixrd/T-U-03/U03-C/3000h",
    data_type="characterisation_report",
)
```

Once all GIXRD records are ingested, a query like *"Which coupons retained
austenite after 3 000 h and what were their corresponding salt-Cr concentrations?"*
draws on both the GIXRD records and the ICP-OES records in one answer.

---

## 8 — Cross-Experiment Analysis: Querying the Full 18-Test Dataset

**Paper connection:** The paper's main finding is the ~33× difference in
corrosion depth and ~194× difference in mass loss between untreated and
purified salt, and the saturation of corrosion depth in untreated salt between
2 000 and 3 000 h.  This is derived by cross-referencing results from all 18
tests.

**Data-layer capability:** Once all ICP-OES, mass-change, SEM, and GIXRD
records are ingested, the RAG pipeline can synthesise cross-test comparisons in
plain language.

```python
answer = rag.answer(
    "Summarise the chromium depletion depth and dissolved Cr concentration "
    "in the salt for all 316L SS coupons tested in FLiNaK, grouped by "
    "salt condition (purified vs. untreated) and exposure time. "
    "Does the untreated-salt corrosion depth appear to plateau after 2000 h?"
)
print(answer)
```

```python
answer = rag.answer(
    "Compare the GIXRD phases found in purified vs. untreated FLiNaK coupons. "
    "Which phases are unique to untreated-salt coupons and which are unique "
    "to purified-salt coupons? What mechanistic interpretation does this support?"
)
```

This supports the paper's Discussion section — and also supports future
researchers replicating or extending the work, who need to understand whether
new data points are consistent with the existing dataset.

---

## 9 — Future Work: Contextualising UF₃/UF₄ and Fission-Product Extensions

**Paper connection:** Section 6 (Conclusion) explicitly lists four directions
for future work: radiation effects, fission-product chemistry, **UF₃/UF₄
additions**, and temperature/flow gradients.  These represent the next
experimental programme.

**Data-layer capability:** The same data layer serves as the foundation for the
follow-on programme.  ORNL archive reports on uranium-bearing salts (MSRE ran
with UF₄ dissolved in FLiBe) and OpenAlex papers on fission-product speciation
are already in the knowledge base.

```python
# Before designing the UF₄-doped FLiNaK tests:
answer = rag.answer(
    "What effect did UF₄ additions have on the corrosion rate of structural "
    "alloys in the MSRE? What U/U4+ redox ratio was maintained, and how was "
    "it controlled? Were there 316 SS or stainless steel tests in uranium-bearing salts?"
)

answer = rag.answer(
    "What tellurium and cesium speciation data exists for FLiNaK at 600 °C "
    "in the ORNL reports? How were fission-product impurities handled "
    "during the MSRE purification cycles?"
)
```

This gives the Copenhagen Atomics team a head-start on experimental design for
the follow-on UF₄ and fission-product tests — grounded in six decades of ORNL
operational data — before a single new test begins.

---

## Summary: Lucas et al. (2025) Experimental Workflow × Data-Layer Capability

| Experimental step (paper section) | Data ingested | Data-layer capability |
|---|---|---|
| Design — ORNL baseline (§1 Intro) | ORNL MSRE/MSBR reports | `rag.load_msr_archive()` |
| Design — recent literature (§1 Intro) | OpenAlex 316L/FLiNaK papers | `python msr_kb_sources.py --update-openalex` |
| During — furnace conditions (§2.1) | Temp, Ar pressure, glovebox O₂ per test | `loader.ingest_sensor_snapshot()` |
| During — salt prep records (§2.1) | Batch IDs, purity assay, impurity levels | `loader.ingest_text()` (salt_preparation_record) |
| Post-test — ICP-OES (§4.3) | Dissolved Cr/Fe/Ni per test tube per time-point | `loader.ingest_text()` (characterisation_report) |
| Post-test — mass change (§4.5) | Pre/post mass, specific mass loss per coupon | `loader.ingest_text()` (characterisation_report) |
| Post-test — SEM depth (§4.4) | Max and mean corrosion depth, attack mode | `loader.ingest_text()` (characterisation_report) |
| Post-test — GIXRD phases (§4.6) | Phase names, mechanistic interpretation | `loader.ingest_text()` (characterisation_report) |
| Analysis — cross-test comparison (§5) | Full 18-test dataset | `rag.answer()` |
| Future work — UF₄ / fission products (§6) | ORNL uranium-salt + fission-product data | `rag.answer()` over existing archive |
