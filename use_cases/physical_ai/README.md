# MSR Data Layer — Physical AI Foundation-Model Training Use Cases

This subfolder documents how the MSR data layer (`msr_data_layer`) supplies
the **training data, retrieval context, and operational records** needed to
develop and fine-tune **foundation models** for each of the twelve robotic
operational areas defined in the
[MSR Physical AI Layer](https://github.com/pranavkantgaur/msr_physical_ai_layer).

Each file maps one physical AI task to the concrete data streams that feed
foundation-model training — sensor time-series, event logs, maintenance
records, ORNL archival knowledge, and simulation outputs — and shows which
data-layer API calls produce them.

---

## Operational Areas (priority order)

| Rank | File | Robot | Area |
|------|------|-------|------|
| 1 | [01_primary_loop_maintenance_repair.md](01_primary_loop_maintenance_repair.md) | PLMR-01 | Primary loop maintenance & repair |
| 2 | [02_hot_cell_chemical_processing.md](02_hot_cell_chemical_processing.md) | HCPR-01 | Hot-cell chemical processing automation |
| 3 | [03_salt_sampling_analysis.md](03_salt_sampling_analysis.md) | SSR-01 | Salt sampling & analysis |
| 4 | [04_radiation_mapping_inspection.md](04_radiation_mapping_inspection.md) | RMR-01 | Radiation mapping & autonomous inspection |
| 5 | [05_freeze_plug_safety_monitoring.md](05_freeze_plug_safety_monitoring.md) | FPMR-01 | Freeze plug safety monitoring |
| 6 | [06_fuel_salt_transport_refilling.md](06_fuel_salt_transport_refilling.md) | FSTR-01 | Fuel salt transport & refilling |
| 7 | [07_graphite_moderator_inspection.md](07_graphite_moderator_inspection.md) | GIR-01 | Graphite moderator inspection & replacement |
| 8 | [08_tritium_management.md](08_tritium_management.md) | TMR-01 | Tritium management systems |
| 9 | [09_off_gas_system_handling.md](09_off_gas_system_handling.md) | OGSR-01 | Off-gas system handling |
| 10 | [10_waste_salt_handling_solidification.md](10_waste_salt_handling_solidification.md) | WSHR-01 | Waste salt handling & solidification |
| 11 | [11_external_structural_inspection.md](11_external_structural_inspection.md) | SIR-01 | External structural inspection |
| 12 | [12_security_safeguards_monitoring.md](12_security_safeguards_monitoring.md) | SPR-01 | Security & safeguards monitoring |

---

## Common Pattern: Data Layer → Foundation Model Training Pipeline

Each use case follows the same three-stage pattern:

```
1. KNOWLEDGE RETRIEVAL
   rag.load_msr_archive()
   rag.answer(question)          ← grounding prompts with ORNL/literature context

2. TRAINING DATA INGESTION
   loader.ingest_sensor_snapshot(rag, readings, source_id=...)
   loader.ingest_text(rag, text=..., source_id=..., data_type=...)

3. DATASET EXPORT FOR FINE-TUNING
   rag.answer(structured_query)  ← retrieve labelled episodes for offline training
```

Foundation models trained on this data gain:
* **domain grounding** — ORNL MSR archive provides physics priors
* **operational context** — live sensor streams provide state representations
* **labelled episodes** — event logs provide reward / outcome labels for RL

---

## Common Data-Layer APIs

| Operation | API |
|---|---|
| Load ORNL MSR archive | `rag.load_msr_archive()` |
| Query knowledge base | `rag.answer(question)` |
| Log robot sensor readings | `loader.ingest_sensor_snapshot(rag, readings, source_id=...)` |
| Store maintenance / procedure records | `loader.ingest_text(rag, text=..., data_type="characterisation_report")` |
| Store event logs | `loader.ingest_text(rag, text=..., data_type="event_log")` |

See [../README.md](../README.md) for installation, and
[../../00_MCP_START_HERE.md](../../00_MCP_START_HERE.md) for the five-minute
quick start.
