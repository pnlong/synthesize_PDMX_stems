# spdmx

Turn the [PDMX](https://zenodo.org/records/13763756) symbolic music dataset into audio stems, captions, and SA3-realified audio.

## Pipeline

1. **Synthesis** — `python -m synthesis.synthesize` with `--render-mode {basic,slakh}`
2. **Realify** (optional) — same command with `--realify`
3. **Full dataset** — `python -m synthesis.build_spdmx` (planned; calls `synthesize --full` internally)
4. **Analysis** — duration stats and SA3 model recommendation

## Install

**→ Full step-by-step guide: [`SETUP.md`](SETUP.md)**

Quick start (synthesis + analysis only):

```bash
cd ~/spdmx
uv sync --group dev
uv run python -c "import mido, synthesis.audio; print('spdmx ok')"
```

For SA3 realify, submodule, flash-attention, and Hugging Face login, follow **Track B** in [`SETUP.md`](SETUP.md).

## Usage

Default output root: `/deepfreeze/pnlong/SPDMX` (`OUTPUT_DIR` in [`shared/config.py`](shared/config.py)).

Development artifacts (ablations, analysis, interim full stems) live under `{OUTPUT_DIR}/dev/`. The assembled dataset goes to `{OUTPUT_DIR}/SPDMX/` via `build_spdmx`.

### Ablation (four conditions)

Default behavior: random sample from `subset:rated_deduplicated` (N=100, seed=42).

```bash
# A1 / B1 — raw stems
uv run python -m synthesis.synthesize --render-mode basic
uv run python -m synthesis.synthesize --render-mode slakh

# Prototyping: MP3 instead of FLAC (smaller; use same flag for realify)
uv run python -m synthesis.synthesize --render-mode basic --mp3
uv run python -m synthesis.synthesize --render-mode basic --mp3 --realify

# A2 / B2 — realify (GPU only; requires A1 / B1 stems first)
uv run python -m synthesis.synthesize --render-mode basic --realify
uv run python -m synthesis.synthesize --render-mode slakh --realify
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
uv run python -m synthesis.synthesize --render-mode basic --full
uv run python -m synthesis.synthesize --render-mode basic --full --realify
```

Output: `{OUTPUT_DIR}/dev/stems/` and `{OUTPUT_DIR}/dev/stems_realify/`.

### Assembled sPDMX dataset (planned)

```bash
uv run python -m synthesis.build_spdmx --render-mode basic
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
uv run python -m analysis.analyze_song_lengths
```

Writes to `{OUTPUT_DIR}/dev/analysis/song_lengths/`:

- `song_length_histogram.png` — distribution with SA3 limits marked
- `song_length_percentiles.png` — empirical CDF (percentile curve)
- `song_length_report.json` — stats, duration percentiles, SA3 limit percentiles, and model recommendation

Also symlinks in-repo dev output (both gitignored; run `uv run python -m shared.setup_symlinks` after clone):

- [`analysis/output/`](analysis/output/) → `{OUTPUT_DIR}/dev/analysis/`
- [`synthesis/ablations_output/`](synthesis/ablations_output/) → `{OUTPUT_DIR}/dev/ablations/`

## Repository layout

| Path | Purpose |
|------|---------|
| [`SETUP.md`](SETUP.md) | **Environment setup guide** (uv, SA3, flash-attn) |
| `synthesis/synthesize.py` | Main CLI: ablation sample (default) or `--full` PDMX |
| `synthesis/build_spdmx.py` | Assemble complete sPDMX dataset (stub) |
| `synthesis/realify/` | SA3 wrapper + submodule |
| `synthesis/realify/captions/` | Caption generation from PDMX metadata |
| `analysis/` | Duration analysis and SA3 model recommendation — see [`analysis/README.md`](analysis/README.md) |
| `shared/config.py` | Paths, ablation sample size, constants — see [`shared/README.md`](shared/README.md) |
| `shared/setup_symlinks.py` | Create in-repo symlinks after clone (`python -m shared.setup_symlinks`) |

See [`synthesis/RENDERING_NOTES.md`](synthesis/RENDERING_NOTES.md) for Slakh alignment, ablation design, and listening test plans. Synthesis layout: [`synthesis/README.md`](synthesis/README.md).

## Tests

```bash
uv run PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p pytest
```

## License

See [LICENSE](LICENSE).
