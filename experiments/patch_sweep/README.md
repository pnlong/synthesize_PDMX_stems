# Patch Pool Sweep

**Step-by-step runbook:** [`GUIDE.md`](GUIDE.md) — start here.

Find the best Slakh-style rendering recipe for realistic per-instrument Fluidsynth synthesis (B1 ablation).

**Methodology:** [`../TUNING.md`](../TUNING.md)  
**Soundfont catalog:** [`soundfonts.yaml`](soundfonts.yaml)

## What we're tuning

Slakh2100 randomizes professional sample patches per GM class (~187 patches, 34 classes). We approximate that with Fluidsynth:

1. **Soundfont** — which GM sample bank
2. **Program pools** — class-matched GM program remapping ([`synthesis/patches.py`](../../synthesis/patches.py))
3. **FX** — light reverb/EQ (after soundfont is chosen)

**Per-category winners** — not one global soundfont or pool. See [`../probe_stems.yaml`](../probe_stems.yaml) categories.

## Staged plan

| Phase | Compare | Grid / config | Status |
|-------|---------|---------------|--------|
| **1 — Soundfonts** | 7 banks, dry, passthrough programs | [`soundfonts.yaml`](soundfonts.yaml) | **Current** |
| **2 — FX** | Light reverb/EQ on 1–3 shortlisted banks | TBD (`fx_grid.yaml`) | Pending |
| **3 — Program pools** | GM program lists per class | [`patch_grid.yaml`](patch_grid.yaml) + `PATCH_POOLS` | Pending |

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

Machine-readable list: [`soundfonts.yaml`](soundfonts.yaml). Update `phase1_status` and `winners` there after audition.

### Phase 1 audition commands

Render one probe stem per bank (smoke):

```bash
uv run python -m experiments.patch_sweep.sweep \
  --soundfont "soundfonts/GeneralUser GS v1.471.sf2" \
  --limit-stems 1 --limit-variants 1 --mp3
```

Full phase-1 matrix (7 banks × 24 stems = 168 renders) — run once per bank, reusing `--output-dir` with distinct variant folders or separate output roots (TBD when we wire soundfont into the grid).

## Program pool grid (phase 3)

| Dimension | Values |
|-----------|--------|
| `pool_id` | `pool_v1_conservative`, `pool_v2_diverse`, `pool_v3_slakh_like` |

3 variants × 24 probe stems ≈ 72 fluidsynth renders (CPU), after pools are populated in `PATCH_POOLS`.

## Prerequisites

1. A1 basic ablation stems exist (`dev/ablations/basic/`)
2. Soundfont symlink: `soundfonts/` in repo root
3. For phase 3: patch pools defined in [`synthesis/patches.py`](../../synthesis/patches.py)

Until pools are populated, the sweep still runs but variants render with passthrough programs.

## Probe set

Shared [`experiments/probe_stems.yaml`](../probe_stems.yaml).

## Run

```bash
# Full pool sweep (phase 3; CPU)
uv run python -m experiments.patch_sweep.sweep -j 8

# Explicit soundfont (phase 1)
uv run python -m experiments.patch_sweep.sweep \
  --soundfont "soundfonts/SGM-V2.01.sf2" -j 8

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

1. Note phase-1 soundfont picks in [`soundfonts.yaml`](soundfonts.yaml) (`winners` section)
2. After listening test, update [`results_notes.md`](results_notes.md)
3. Aggregate: `uv run python -m experiments.listening.aggregate --sweep patch ...`
4. Lock winning pools in `synthesis/patches.py` → run B1 ablation
