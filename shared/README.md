# shared

Cross-cutting configuration and constants used by synthesis and analysis.

## Files

| File | Description |
|------|-------------|
| `config.py` | Paths (`OUTPUT_DIR`, `PDMX_FILEPATH`), ablation sample settings, table column schemas, audio/render constants, SA3 duration limits |

Key paths (see `config.py` for full list):

- `OUTPUT_DIR` — `/deepfreeze/pnlong/SPDMX`
- `{OUTPUT_DIR}/dev/` — development artifacts (ablations, analysis, interim stems)
- `{OUTPUT_DIR}/SPDMX/` — final assembled dataset (via `build_spdmx`)
