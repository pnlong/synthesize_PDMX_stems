# SA3 Preset Sweep

Find the best `init_noise_level` and prompt format for SA3 realify: realistic timbre while preserving melody, rhythm, and timing.

## Grid

| Dimension | Values |
|-----------|--------|
| `init_noise_level` | 0.25, 0.35, 0.45, 0.55, 0.65 |
| `prompt_variant` | `current`, `minimal`, `preservation` |

15 variants × 15 probe stems ≈ 225 SA3 forwards.

## Probe set

[`probe_stems.yaml`](probe_stems.yaml) — curated stems from `dev/ablations/basic/` covering piano, drums, strings, wind, voice, mallet, organ, and polyphonic cases.

## Run

```bash
# Full sweep (GPU)
uv run python -m experiments.preset_sweep.sweep

# Smoke test
uv run python -m experiments.preset_sweep.sweep --limit-stems 1 --limit-variants 2

# MP3 (smaller files; must match source ablation format)
uv run python -m experiments.preset_sweep.sweep --mp3
```

Outputs go to `{OUTPUT_DIR}/dev/experiments/preset_sweep/` (symlinked at [`output/`](output/)).

## Listen

```bash
uv run python -m synthesis.listening.serve --preset-sweep
```

Open [http://127.0.0.1:8765/preset-sweep](http://127.0.0.1:8765/preset-sweep) — stem-centric grid comparing A1 (raw) vs each variant.

## Record results

After listening, note per-instrument winners in [`results_notes.md`](results_notes.md), then update [`synthesis/realify/presets/categories.yaml`](../../synthesis/realify/presets/categories.yaml).

## Listening protocol

For each probe stem, compare A1 vs all variants on:

1. **Content** — same melody, rhythm, timing?
2. **Realism** — sounds like a recorded instrument?

Expect different winners per instrument class.
