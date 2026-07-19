# Patch Sweep

**Step-by-step runbook:** [`GUIDE.md`](GUIDE.md) — start here.

Find the best Slakh-style rendering recipe for realistic per-instrument Fluidsynth synthesis (B1 ablation).

**Methodology:** [`../TUNING.md`](../TUNING.md)  
**Soundfont catalog:** [`soundfonts.yaml`](soundfonts.yaml)

## What we're tuning

Slakh2100 uses diverse professional sample patches per GM class. We approximate that with Fluidsynth:

1. **Soundfont shortlists** — multiple GM banks per listening category (mean rating ≥ 4.1)
2. **FX** — light reverb/EQ (after soundfont is chosen)

Production slakh mode randomly picks a soundfont per (song, category) from each shortlist. MIDI programs are unchanged.

**Per-category settings** — not one global soundfont. See [`../probe_stems.yaml`](../probe_stems.yaml) categories.

## Staged plan

| Phase | Compare | Grid / config | Status |
|-------|---------|---------------|--------|
| **1 — Soundfonts** | 7 banks, dry, passthrough programs | [`grids/phase1_soundfonts.yaml`](grids/phase1_soundfonts.yaml) | **Current** |
| **2 — FX** | Light reverb/EQ on phase-1 shortlists | [`grids/phase2_fx.yaml`](grids/phase2_fx.yaml) | Pending |
| **3 — Lock** | Shortlists + FX → `winners_locked.yaml` | `lock` | Pending |

Do not combine all dimensions in one listening session until earlier phases are decided.

## Soundfonts in use

Stored at `/data3/pnlong/soundfonts`, symlinked in-repo as [`soundfonts/`](../../soundfonts/) (`uv run python -m shared.setup_symlinks`).

| ID | File | ~MB | Notes |
|----|------|-----|-------|
| `sgm_v2` | `SGM-V2.01.sf2` | 236 | **Production default** (`SOUNDFONT_PATH`) |
| `generaluser` | `GeneralUser GS v1.471.sf2` | 30 | Reddit/archive favorite; light and musical |
| `fluidr3` | `FluidR3_GM2-2.SF2` | 141 | Standard GM reference |
| `arachno` | `Arachno_SoundFont_Version_1.0.sf2` | 149 | Strings / brass / bass |
| `airfont_380` | `airfont_380_final.sf2` | 264 | Bright winds, piano |
| `realfont` | `RealFont_2_1.sf2` | 102 | Balanced; punchy drums |
| `timgm6mb` | `TimGM6mb.sf2` | 6 | Flakh2100 baseline (sanity check only) |

Machine-readable list: [`soundfonts.yaml`](soundfonts.yaml).

## Prerequisites

1. A1 basic ablation stems exist (`dev/ablations/basic/`)
2. Soundfont symlink: `soundfonts/` in repo root

## Probe set

Shared [`experiments/probe_stems.yaml`](../probe_stems.yaml).

## Run

```bash
# Phase 1 soundfonts
uv run python -m experiments.patch_sweep.sweep \
  --phase phase1_soundfonts -j 8

# Phase 2 FX (after phase 1 recorded)
uv run python -m experiments.patch_sweep.sweep \
  --phase phase2_fx -j 8

# Smoke test
uv run python -m experiments.patch_sweep.sweep --limit-stems 1 --limit-variants 1

# MP3 (smaller files)
uv run python -m experiments.patch_sweep.sweep --mp3 -j 4
```

Outputs go to `{OUTPUT_DIR}/dev/experiments/patch_sweep/` (symlinked at [`output/`](output/)).

## Listen

```bash
uv run python -m experiments.listening.serve --sweep patch
```

Open [http://127.0.0.1:8766/test?type=patch](http://127.0.0.1:8766/test?type=patch). Reference anchor = A1 basic stem.

## Record results

```bash
# Phase 1: shortlists (mean rating >= 4.1)
uv run python -m experiments.patch_sweep.record_winners \
  --phase phase1_soundfonts \
  --responses experiments/patch_sweep/output/phase1_soundfonts/responses/responses_....json

# Phase 2: single FX winner per category
uv run python -m experiments.patch_sweep.record_winners \
  --phase phase2_fx \
  --responses experiments/patch_sweep/output/phase2_fx/responses/responses_....json

# Lock production config
uv run python -m experiments.patch_sweep.lock
```

Then run B1 ablation: `uv run python -m synthesis.synthesize --render-mode slakh`
