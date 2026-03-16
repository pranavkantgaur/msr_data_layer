# MSR Data Layer — Use-Case Case Studies

This directory contains case studies showing how the MSR data layer
(`msr_data_layer`) supports concrete experimental research and reactor-operations
work drawn from published MSR science.

Each file maps a specific high-impact paper or technical report to a
data-layer capability, providing:
* a summary of the experimental work and its key data streams,
* numbered subsections linking each data stream to a data-layer API
  (`rag.answer()`, `ingest_sensor_snapshot()`, `ingest_text()`), and
* runnable Python/CLI code examples.

---

## Case Studies

| File | Paper / Report | Topic | Year |
|---|---|---|---|
| [lucas_2025_316L_flinak_corrosion.md](lucas_2025_316L_flinak_corrosion.md) | Lucas N. et al., *J. Nucl. Mater.* | 316L SS corrosion in purified vs. untreated FLiNaK and LiThF | 2025 |
| [haubenreich_engel_1970_msre_operations.md](haubenreich_engel_1970_msre_operations.md) | Haubenreich & Engel, *Nucl. Appl. Technol.* | Complete MSRE operational experience (13 172 h) | 1970 |
| [koger_1972_hastelloy_n_corrosion.md](koger_1972_hastelloy_n_corrosion.md) | Koger, ORNL-TM-4273 | Corrosion and mass transfer of Hastelloy N in molten fluorides | 1972 |
| [cantor_1968_fluoride_salt_properties.md](cantor_1968_fluoride_salt_properties.md) | Cantor et al., ORNL-4229 | Physical properties of MSR fuel, coolant and flush salts | 1968 |
| [mccoy_1970_tellurium_embrittlement.md](mccoy_1970_tellurium_embrittlement.md) | McCoy et al., *Nucl. Appl. Technol.* | Tellurium-induced intergranular cracking of Hastelloy N | 1970 |
| [baes_1974_redox_chemistry.md](baes_1974_redox_chemistry.md) | Baes, *J. Nucl. Mater.* | Redox chemistry and UF₃/UF₄ control in FLiBe fuel salt | 1974 |
| [bettis_robertson_1970_thorium_breeding_fuel_cycle.md](bettis_robertson_1970_thorium_breeding_fuel_cycle.md) | Bettis & Robertson, *Nucl. Appl. & Tech.* | Thorium breeding cycle, Pa-233 management, and online fuel processing for Copenhagen Atomics' WATT reactor | 1970 |
| [nrc_2017_msr_regulatory_challenges.md](nrc_2017_msr_regulatory_challenges.md) | NRC Staff, ADAMS ML17331B126 | Regulatory challenges for MSR licensing — LBEs, source term, materials qualification, QA records, safeguards | 2017 |
| [xue_2026_prhrs_transient_evaluation.md](xue_2026_prhrs_transient_evaluation.md) | Xue S. et al., *Ann. Nucl. Energy* | Transient performance evaluation of passive residual heat removal system in liquid molten salt reactor (RELAP5-TMSR) | 2026 |
| [yang_2025_msp_pump_thermal_optimisation.md](yang_2025_msp_pump_thermal_optimisation.md) | Yang J. et al., SINAP/CAS | Thermal resistance optimisation of high-temperature molten salt pump via RSM and NSGA-II (10.3 °C bearing temperature reduction) | 2025 |
| [wang_2026_resta3d_tmsr_transient_safety.md](wang_2026_resta3d_tmsr_transient_safety.md) | Wang K. et al., SINAP/CAS, *Ann. Nucl. Energy* | RESTA-3D reactor dynamics code with DNP transport and N-TH coupling — validation against MSRE and 2 MWth TMSR transient safety analysis | 2026 |

---

## Common Data-Layer APIs Used in All Case Studies

| Operation | API |
|---|---|
| Load ORNL MSR technical report archive | `rag.load_msr_archive()` |
| Update OpenAlex academic literature | `python msr_kb_sources.py --update-openalex` |
| Query knowledge base in plain language | `rag.answer(question)` |
| Log periodic sensor / instrument readings | `loader.ingest_sensor_snapshot(rag, readings, source_id=...)` |
| Store characterisation or analysis results | `loader.ingest_text(rag, text=..., source_id=..., data_type=...)` |
| Store event logs (maintenance, additions) | `loader.ingest_text(rag, text=..., data_type="event_log")` |

See [README.md](../README.md) for installation and setup instructions,
and [00_MCP_START_HERE.md](../00_MCP_START_HERE.md) for the five-minute
quick start.

---

## Adding New Case Studies

To add a case study for a new paper:

1. Create a new file `use_cases/<author>_<year>_<short_topic>.md`.
2. Use the existing files as templates — each section should have a
   **Paper connection** paragraph and a **Data-layer capability** paragraph
   with runnable code examples.
3. Add a row to the table above.

The case study does not need to reference real experiment IDs or timestamps;
illustrative values (as used throughout this directory) are sufficient to
demonstrate the API usage pattern.
