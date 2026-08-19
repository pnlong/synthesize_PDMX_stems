# shared

Cross-cutting configuration and constants used by synthesis and analysis.

## Files

| File | Description |
|------|-------------|
| `config.py` | Paths (`OUTPUT_DIR`, `PDMX_FILEPATH`), ablation sample settings, table column schemas, audio/render constants, SA3 duration limits |
| `repo_symlinks.py` | In-repo symlink helpers for `analysis/output/` and `synthesis/ablations_output/` |
| `setup_symlinks.py` | CLI: `python -m shared.setup_symlinks` — run after clone on a new machine |

Key paths (see `config.py` for full list):

- `OUTPUT_DIR` — `/deepfreeze/pnlong/SPDMX`
- `{OUTPUT_DIR}/dev/` — development artifacts (ablations, analysis, interim stems)
- `{OUTPUT_DIR}/SPDMX/` — released dataset (`LICENSE`, `README.md`, `track_map.csv`, `audio/`, `mid/`; join to PDMX.csv on `song_id`)
- `{OUTPUT_DIR}/dev/final/` — production synthesis tables (`data.csv`, `stems.csv`, `stem_recipe.csv`)
