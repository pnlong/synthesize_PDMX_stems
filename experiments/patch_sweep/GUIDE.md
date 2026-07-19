# Slakh patch tuning — step-by-step runbook

Follow these steps in order. Each phase has a **blinded listening test** (you only see Sample A/B/C, not soundfont or FX names). Winners chain into the next phase.

**Frozen phases:** Once a phase is listened and recorded in `winners.yaml`, you do not need to re-render it. Updates to `probe_stems.yaml` (e.g. fixing MIDI program mismatches) apply to **later phases only** — existing phase outputs and responses stay valid.

**Prerequisites**

```bash
# Soundfonts symlinked
uv run python -m shared.setup_symlinks

# A1 basic ablation probe stems exist
ls synthesis/ablations_output/basic/data/

# Optional: smaller MP3 outputs for listening
export SWEEP_MP3="--mp3"
```

---

## Phase 1 — Soundfonts (dry, no program remap)

Compare **7 candidate GM banks**. Same MIDI, different samples. Build a **shortlist per category** (not a single winner): every soundfont with mean(content, realism) / 2 **≥ 4.1** is included.

### 1.1 Render

```bash
uv run python -m experiments.patch_sweep.sweep \
  --phase phase1_soundfonts -j 8 $SWEEP_MP3
```

~7 variants × 24 stems = **168** CPU renders.  
Output: `experiments/patch_sweep/output/phase1_soundfonts/`

### 1.2 Blinded listening test

```bash
uv run python -m experiments.listening.serve --sweep patch \
  --patch-sweep-dir experiments/patch_sweep/output/phase1_soundfonts
```

Open [http://127.0.0.1:8766/test?type=patch](http://127.0.0.1:8766/test?type=patch)

- Reference = A1 basic stem (unchanged)
- Rate each blinded sample: **content** + **realism**
- Phase 1 tip: content should be nearly equal; **realism** is the main signal
- Each **Next stem** auto-saves to `responses/responses_in_progress.json` on the server
- **Finish** writes a timestamped `responses/responses_YYYYMMDDTHHMMSSZ.json` and shows the path on screen (no forced browser download)
- **Save to server** in the header forces a checkpoint anytime

### 1.3 Record winners

```bash
uv run python -m experiments.patch_sweep.record_winners \
  --phase phase1_soundfonts \
  --responses experiments/patch_sweep/output/phase1_soundfonts/responses/responses_YYYYMMDDTHHMMSSZ.json
```

This writes per-category **shortlists** (e.g. `piano: [sgm_v2, airfont_380]`) into [`winners.yaml`](winners.yaml). Soundfonts below the threshold are dropped; if none pass, the best-rated one is kept as a fallback.

Default threshold is **4.1** (average of content + realism). Override with `--mean-rating-threshold`.

Optional: inspect aggregate report:

```bash
uv run python -m experiments.listening.aggregate \
  --sweep patch \
  --sweep-dir experiments/patch_sweep/output/phase1_soundfonts \
  --responses experiments/patch_sweep/output/phase1_soundfonts/responses/responses_....json \
  --output experiments/patch_sweep/results_notes.md
```

---

## Phase 2 — FX (on phase-1 shortlists)

Compare **3 light FX profiles** using each category's **primary** phase-1 soundfont (first in the shortlist).

### 2.1 Render

Requires `winners.yaml` phase 1 `completed: true`.

```bash
uv run python -m experiments.patch_sweep.sweep \
  --phase phase2_fx -j 8 $SWEEP_MP3
```

~3 variants × 24 stems = **72** renders.  
Output: `experiments/patch_sweep/output/phase2_fx/`

### 2.2 Listen → record

```bash
uv run python -m experiments.listening.serve --sweep patch \
  --patch-sweep-dir experiments/patch_sweep/output/phase2_fx

uv run python -m experiments.patch_sweep.record_winners \
  --phase phase2_fx \
  --responses experiments/patch_sweep/output/phase2_fx/responses/responses_....json
```

---

## Phase 3 — Soundfont shortlist review (before lock)

After phases 1–2 are recorded, **review the phase 1 shortlist** — listen to each soundfont dry (no FX) and reject anything that slipped through the rating threshold.

```bash
uv run python -m experiments.listening.serve --sweep patch
```

Open [http://127.0.0.1:8766/verify?type=patch](http://127.0.0.1:8766/verify?type=patch), select your **phase 1** `responses_*.json`, then for each category:

- Click soundfont tabs (or use prev/next) to audition each shortlisted bank
- Compare reference (A1 basic) vs dry soundfont on all probe stems
- **Uncheck** soundfonts you do not want in production
- Move to the next category when at least one soundfont remains

**Finish** writes `experiments/patch_sweep/output/phase1_soundfonts/responses/verification_final_responses_*_YYYYMMDDTHHMMSSZ.json`.

---

## Phase 4 — Lock production config

When the shortlist review looks good:

```bash
uv run python -m experiments.patch_sweep.lock \
  --verification experiments/patch_sweep/output/phase1_soundfonts/responses/verification_final_responses_....json
```

(`--verification` updates phase 1 shortlists from your review, then locks.)

Writes [`winners_locked.yaml`](winners_locked.yaml) — per-category:

```yaml
categories:
  piano:
    soundfont_ids: [sgm_v2, airfont_380]
    soundfont_id: sgm_v2
    soundfont: SGM-V2.01.sf2
    fx_profile: light
```

Production slakh mode **randomly picks** a soundfont per (song, category) from each shortlist. MIDI programs are unchanged (no GM pool remapping).

`synthesis/patches.py` loads this automatically as `SLAKH_CATEGORY_RENDER`.

---

## Phase 5 — Validate & run B1 ablation

### 5.1 Sanity check (probe stems)

Re-render a few probe stems in slakh mode and confirm they differ from A1:

```bash
uv run python -m experiments.patch_sweep.sweep \
  --phase phase2_fx --limit-stems 2 --limit-variants 1 $SWEEP_MP3
```

Or run synthesis on a single song if you have a quick test path.

### 5.2 Full B1 ablation

```bash
uv run python -m synthesis.synthesize --render-mode slakh -j 8
```

Output: `dev/ablations/slakh/`

### 5.3 Ablation listening comparison

```bash
uv run python -m synthesis.listening.serve
```

Compare A1 (basic) vs B1 (slakh) on port **8765**.

---

## Quick reference

| Step | Command |
|------|---------|
| Render phase N | `uv run python -m experiments.patch_sweep.sweep --phase <phase> -j 8` |
| Listen (per phase) | `uv run python -m experiments.listening.serve --sweep patch --patch-sweep-dir <phase_dir>` |
| Final verify | `http://127.0.0.1:8766/verify?type=patch` (review phase 1 shortlists) |
| Record winners | `uv run python -m experiments.patch_sweep.record_winners --phase <phase> --responses <json>` |
| Lock production | `uv run python -m experiments.patch_sweep.lock [--verification <json>]` |
| B1 ablation | `uv run python -m synthesis.synthesize --render-mode slakh` |

| Phase | `--phase` value | Variants | Needs prior winners |
|-------|-----------------|----------|---------------------|
| 1 Soundfonts | `phase1_soundfonts` | 7 | — |
| 2 FX | `phase2_fx` | 3 | phase 1 shortlists |

## Files

| File | Purpose |
|------|---------|
| [`grids/`](grids/) | Variant definitions per phase |
| [`soundfonts.yaml`](soundfonts.yaml) | Candidate soundfont catalog |
| [`winners.yaml`](winners.yaml) | Your per-phase decisions (updated by `record_winners`) |
| [`winners_locked.yaml`](winners_locked.yaml) | Production config (written by `lock`) |
| [`results_notes.md`](results_notes.md) | Subjective notes template |

## Troubleshooting

- **Phase 2 sweep refuses to start** — run `record_winners` for phase 1 first; check `winners.yaml` shows `completed: true`.
- **Missing probe stem** — ensure A1 basic ablation exists for all `probe_stems.yaml` song IDs.
- **Soundfont not found** — run `ls soundfonts/`; recreate symlink via `setup_symlinks`.
- **Listening UI shows "Audio not available"** — sweep didn't finish; check manifest.csv `out_path` files exist.
