# SA3 Preset Sweep

Find the best `init_noise_level`, `prompt_variant`, and optional diffusion budget for SA3 realify: realistic timbre while preserving melody, rhythm, and timing (A2 ablation).

**Methodology:** [`../TUNING.md`](../TUNING.md)  
**Step-by-step runbook:** [`GUIDE.md`](GUIDE.md)

## Phased tuning (do not skip ahead)

| Phase | What we compare | Status |
|-------|-----------------|--------|
| **1 — Noise** | 5 `init_noise_level` values, fixed `current` prompt | Pending |
| **2 — Prompts** | 3 prompt variants on phase-1 noise winners | Pending |
| **3 — Diffusion** | 3 steps×cfg profiles (optional) | Pending |
| **4 — Lock** | Per-category winners → `categories.yaml` | Pending |

## What we're tuning

SA3 audio-to-audio realify transforms A1 basic Fluidsynth stems. This sweep finds **per-category** preset overrides — not one global noise level or prompt for all instruments.

| Parameter | Phase | Fixed until phase |
|-----------|-------|-------------------|
| `init_noise_level` | 1 | — |
| `prompt_variant` | 2 | noise from phase 1 |
| `steps` / `cfg_scale` | 3 (optional) | noise + prompt from phases 1–2 |
| SA3 model | — | `medium` (post-trained) |
| Source audio | — | A1 `basic` ablation stems |

Production defaults today (pre-tuning): `init_noise_level: 0.45`, `prompt_variant: current` in [`synthesis/realify/presets/categories.yaml`](../../synthesis/realify/presets/categories.yaml).

### What we are *not* tuning here

- Slakh patch pools / soundfonts (see [`../patch_sweep/`](../patch_sweep/))
- SA3 model choice (`medium` vs `small-music`)
- Realify chunking, batch size, or GPU layout

## Grids

Phase grids live in [`grids/`](grids/). The flat [`preset_grid.yaml`](preset_grid.yaml) (15-variant factorial) is kept for reference.

| Phase | Variants × 24 stems | SA3 forwards |
|-------|---------------------|--------------|
| 1 Noise | 5 | 120 |
| 2 Prompts | 3 | 72 |
| 3 Diffusion | 3 | 72 (optional) |

## Probe set

Shared [`experiments/probe_stems.yaml`](../probe_stems.yaml) — same 24 stems (3 per category) as patch sweep for comparable per-category decisions.

## Prerequisites

1. A1 basic ablation stems complete (`dev/ablations/basic/`)
2. GPU with ≥10 GiB free VRAM per worker (`REALIFY_MIN_GPU_FREE_GB`)
3. SA3 `medium` model available (see [`synthesis/realify/README.md`](../../synthesis/realify/README.md))

## Run

See [`GUIDE.md`](GUIDE.md) for the full workflow. Quick start:

```bash
# Phase 1
uv run python -m experiments.preset_sweep.sweep --phase phase1_noise

# After listening + record_winners for each phase:
uv run python -m experiments.preset_sweep.lock
```

Smoke test:

```bash
uv run python -m experiments.preset_sweep.sweep \
  --phase phase1_noise --limit-stems 1 --limit-variants 2
```

Outputs go to `{OUTPUT_DIR}/dev/experiments/preset_sweep/` (symlinked at [`output/`](output/)).

## Listen

```bash
uv run python -m experiments.listening.serve --sweep preset \
  --preset-sweep-dir experiments/preset_sweep/output/phase1_noise
```

Open [http://127.0.0.1:8766/test?type=preset](http://127.0.0.1:8766/test?type=preset). Reference anchor = A1 raw (unrealified) stem.

## Lock production

```bash
uv run python -m experiments.preset_sweep.lock
```

Writes [`winners_locked.yaml`](winners_locked.yaml) and updates [`categories.yaml`](../../synthesis/realify/presets/categories.yaml).
