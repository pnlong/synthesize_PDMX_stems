# analysis

Tools for understanding PDMX / sPDMX dataset characteristics and choosing a Stable Audio 3 model.

Output symlinked in-repo at [`analysis/output/`](output/) → `{OUTPUT_DIR}/dev/analysis/` (gitignored; created by `shared.setup_symlinks` or `analyze_song_lengths`).

## Which tool do I use?

| Tool | Input | Measures | When to use |
|------|-------|----------|-------------|
| **`analyze_gm_register`** | PDMX MIDI + [`gm_register_aliases.yaml`](gm_register_aliases.yaml) | Per-track GM corrections + stats | **Step 0 before any synthesize ablation.** Required so stems use corrected programs. |
| **`analyze_song_lengths`** | PDMX.csv (`song_length.seconds`) | Song duration (seconds), all valid rows | **Default for SA3 model choice.** Fast, no synthesis required. |
| **`analyze_durations`** | Synthesized dataset dir (`stems.csv` + FLACs) | Per-stem audio duration (seconds) | Validate that rendered stems match metadata; analyze a specific ablation/full run. |
| **`report`** | `duration_analysis.csv` from above | Aggregated stats + breakdowns | Summarize FLAC-based durations (program, genre, drum). Secondary to `analyze_song_lengths` for model choice. |
| **`analyze_stems`** | PDMX MIDI files | Note count per track (symbolic) | Explore synthesis load / `MAX_N_NOTES_IN_STEM` limits. Not a duration analysis. |
| **`analyze_gm_programs`** | PDMX MIDI, or `--from-register` | GM program usage counts + bar chart | Raw MIDI inventory, or **corrected** inventory from `register.csv` (`gm_program_*_corrected.*`). |

**`analyze_gm_register` is a synthesis prerequisite**, not optional exploration: run it (or refresh it after editing the alias YAML) before `synthesis.synthesize`.

**`analyze_song_lengths` vs `analyze_durations`:** both concern time in seconds, but song lengths reads symbolic metadata for the full dataset; analyze durations reads actual audio files from whatever subset you synthesized (e.g. 100-song ablation).

**`analyze_stems` vs the others:** note counts, not seconds. Unrelated to SA3 duration limits.

## Source files

| File | Purpose |
|------|---------|
| `analyze_gm_register.py` | CLI: build GM register CSV + correction stats |
| `gm_register.py` | Name→program matcher, report builders, synthesize lookup helpers |
| `gm_register_aliases.yaml` | Curated mislabeling needles (SATB→voice, harpsichord, …) |
| `analyze_song_lengths.py` | CLI: PDMX song-length analysis, plots, JSON report, repo symlink |
| `analyze_durations.py` | CLI: measure FLAC stem durations in a synthesized dataset |
| `analyze_stems.py` | CLI: MIDI note-count quantiles across PDMX |
| `analyze_gm_programs.py` | CLI: GM program usage inventory |
| `report.py` | CLI: SA3 recommendation + breakdowns from `duration_analysis.csv` |
| `pdmx_lengths.py` | Load `song_length.seconds`, percentile tables, SA3 limit stats |
| `plots.py` | Histogram and empirical CDF plots for song lengths |
| `tests/` | Unit tests |

## Commands

```bash
# Step 0 — GM register (required before any ablation)
python -m analysis.analyze_gm_register --subset all_valid -j 8
# → {OUTPUT_DIR}/dev/analysis/instruments/all_valid/register.csv
# Re-print stats from an existing register:
python -m analysis.analyze_gm_register --from-csv .../register.csv

# SA3 model selection (recommended)
python -m analysis.analyze_song_lengths

# Optional: validate rendered ablation stems
python -m analysis.analyze_durations --dataset_dir .../dev/ablations/basic
python -m analysis.report

# Optional: symbolic note-count exploration
python -m analysis.analyze_stems

# GM program inventory (raw MIDI)
python -m analysis.analyze_gm_programs

# Corrected GM inventory from register (also written automatically by analyze_gm_register)
python -m analysis.analyze_gm_programs --from-register
# → gm_program_counts_corrected.png / gm_program_report_corrected.json / gm_program_stems_corrected.csv
# → gm_program_counts_compare.png (two-panel original vs corrected, if gm_program_stems.csv exists)
```

See [`output/README.md`](output/README.md) for generated files.
