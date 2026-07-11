# spdmx

Turn the [PDMX](https://zenodo.org/records/13763756) symbolic music dataset into audio stems, captions, and SA3-realified audio.

## Pipeline

1. **Synthesis** — `python -m synthesis.synthesize` with `--render-mode {basic,slakh}`
2. **Realify** (optional) — same command with `--realify`
3. **Full dataset** — `python -m synthesis.build_spdmx` (planned; calls `synthesize --full` internally)
4. **Analysis** — duration stats and SA3 model recommendation

## Install

Uses [uv](https://docs.astral.sh/uv/). See [`synthesis/realify/SETUP.md`](synthesis/realify/SETUP.md) for full environment + SA3 setup.

### Synthesis and analysis only

```bash
uv sync --group dev
```

System: **fluidsynth** must be on `PATH` (see SETUP.md).

### With Stable Audio 3 realify

```bash
git submodule update --init synthesis/realify/stable-audio-3
uv sync --extra realify --group dev
# flash-attention + huggingface-cli login — see synthesis/realify/SETUP.md
```

Verify:

```bash
uv run python -m analysis.analyze_song_lengths
uv run PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest -q
```

## Usage

Default output root: `/deepfreeze/pnlong/SPDMX` (`OUTPUT_DIR` in [`shared/config.py`](shared/config.py)).

Development artifacts (ablations, analysis, interim full stems) live under `{OUTPUT_DIR}/dev/`. The assembled dataset goes to `{OUTPUT_DIR}/SPDMX/` via `build_spdmx`.

### Ablation (four conditions)

Default behavior: random sample from `subset:rated_deduplicated` (N=100, seed=42).

```bash
# A1 / B1 — raw stems
python -m synthesis.synthesize --render-mode basic
python -m synthesis.synthesize --render-mode slakh

# A2 / B2 — realify (uses existing stems or synthesizes first)
python -m synthesis.synthesize --render-mode basic --realify
python -m synthesis.synthesize --render-mode slakh --realify
```

Output:

```
/deepfreeze/pnlong/SPDMX/dev/ablations/
├── basic/
├── basic_realify/
├── slakh/
└── slakh_realify/
```

### Full PDMX (after listening test)

```bash
python -m synthesis.synthesize --render-mode basic --full
python -m synthesis.synthesize --render-mode basic --full --realify
```

Output: `{OUTPUT_DIR}/dev/stems/` and `{OUTPUT_DIR}/dev/stems_realify/`.

### Assembled sPDMX dataset (planned)

```bash
python -m synthesis.build_spdmx --render-mode basic
```

Will populate `{OUTPUT_DIR}/SPDMX/` with PDMX metadata plus synthesized stems in one pass.

### Per-song layout

```
{dataset}/
├── data.csv
├── stems.csv
└── data/<mirrored-path>/
    ├── stem_0.flac
    └── mixture.flac   # sum of stems with uniform anti-clip gain (Slakh-style)
```

### Analysis

Song-length analysis uses PDMX metadata (`song_length.seconds`) — no synthesis required:

```bash
python -m analysis.analyze_song_lengths
```

Writes to `{OUTPUT_DIR}/dev/analysis/song_lengths/`:

- `song_length_histogram.png` — distribution with SA3 limits marked
- `song_length_percentiles.png` — empirical CDF (percentile curve)
- `song_length_report.json` — stats, duration percentiles, SA3 limit percentiles, and model recommendation

Also symlinks [`analysis/output/`](analysis/output/) in this repo to `{OUTPUT_DIR}/dev/analysis/` (report, plots, and future analysis artifacts).

## Repository layout

| Path | Purpose |
|------|---------|
| `synthesis/synthesize.py` | Main CLI: ablation sample (default) or `--full` PDMX |
| `synthesis/build_spdmx.py` | Assemble complete sPDMX dataset (stub) |
| `synthesis/realify/` | SA3 wrapper + submodule |
| `synthesis/realify/captions/` | Caption generation from PDMX metadata |
| `analysis/` | Duration analysis and SA3 model recommendation — see [`analysis/README.md`](analysis/README.md) |
| `shared/config.py` | Paths, ablation sample size, constants — see [`shared/README.md`](shared/README.md) |

See [`synthesis/RENDERING_NOTES.md`](synthesis/RENDERING_NOTES.md) for Slakh alignment, ablation design, and listening test plans. Synthesis layout: [`synthesis/README.md`](synthesis/README.md).

## Tests

```bash
uv run PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest
```

## License

See [LICENSE](LICENSE).
