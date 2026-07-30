# Slakh patch tuning — step-by-step runbook

Follow these steps in order. Each phase has a **blinded listening test** (you only see Sample A/B/C, not soundfont or FX names). Winners chain into the next phase.

**Frozen phases:** Once a phase is listened and recorded in `winners.yaml`, you do not need to re-render it. Updates to `probe_stems.yaml` (e.g. fixing MIDI program mismatches) apply to **later phases only** — existing phase outputs and responses stay valid.

**Prerequisites**

```bash
# Soundfonts symlinked
uv run python -m shared.setup_symlinks

# A1 basic ablation probe stems exist
ls synthesis/ablations_output/basic/data/
```

Stems default to **MP3** (smaller, faster for listening sweeps). Pass `--flac` on sweep/synthesis commands for lossless FLAC.

---

## Phase 1 — Soundfonts (recommended: full archive)

Compare **all 90 archive GM banks** per category — no tag filtering, so sneaky good fonts aren't missed. Same MIDI across variants; only timbre changes.

Build a **shortlist per category** from swipe votes (4 tiers). Every soundfont with at least one **strong accept** clip is included; if none, up to **3 weak accepts** per category.

### 1.1 Render (priority categories first)

```bash
# piano + voice + strings: 90 soundfonts × 9 stems ≈ 810 renders
uv run python -m experiments.patch_sweep.sweep \
  --phase phase1_archive_soundfonts \
  --categories piano,voice,strings \
  -j 8
# Or all 24 probe stems: 90 × 24 ≈ 2160 renders
uv run python -m experiments.patch_sweep.sweep \
  --phase phase1_archive_soundfonts -j 8```

Output: `experiments/patch_sweep/output/phase1_archive_soundfonts/`

Use `--limit-variants 10` for a smoke test. Archive candidates come from [`archive_soundfonts.yaml`](archive_soundfonts.yaml) automatically.

### 1.2 Build 10s clips

Slice each rendered stem into **3 aligned 10-second windows** per probe (chosen from the basic reference for content density):

```bash
uv run python -m experiments.patch_sweep.make_clips \
  --sweep-dir experiments/patch_sweep/output/phase1_archive_soundfonts \
  --categories piano,voice,strings \
  --clips-per-stem 3 \
  -j 8
```

Output: `{sweep_dir}/clips/variants/...` plus `clip_manifest.yaml` and `clip_manifest.csv`.

Re-run with `--force` to rebuild clips after changing window selection.

### 1.3 Swipe listening test (recommended)

```bash
uv run python -m experiments.listening.serve --sweep patch \
  --patch-sweep-dir experiments/patch_sweep/output/phase1_archive_soundfonts
```

Open [http://127.0.0.1:8766/swipe?type=patch](http://127.0.0.1:8766/swipe?type=patch) (all categories in one session).

- One card = one **10s clip** from one blinded soundfont
- Clips **auto-play**; use arrow keys only after the first click (browser autoplay policy)
- `←` strong reject · `→` strong accept · `↓` weak reject · `↑` weak accept
- `Space` replay · `Backspace` undo last vote
- Work through **all categories in one session** by default (no `category` param)
- Optional: `?category=piano` to limit to one listening category
- Default order is shuffled (`?order=shuffle&seed=42`); already-voted cards are skipped on resume (by stable `card_id`, not queue index)
- Progress saves after **every swipe** to `responses/swipe_in_progress.json`; resume across sessions
- **Finish** on the last card writes `responses/swipe_YYYYMMDDTHHMMSSZ.json`

Legacy star-rating UI (`/test?type=patch`) remains for the 7-bank phase 1 path.

### 1.4 Record shortlists

```bash
uv run python -m experiments.patch_sweep.record_winners \
  --phase phase1_archive_soundfonts \
  --mode swipe \
  --responses experiments/patch_sweep/output/phase1_archive_soundfonts/responses/swipe_YYYYMMDDTHHMMSSZ.json
```

Writes per-category shortlists into [`winners.yaml`](winners.yaml) under the `phase1_soundfonts` key (what phase 2 reads).

Winner rules (default):

- Include **all** variants with ≥1 **strong accept** clip vote
- Else include up to **3** variants with **weak accept** votes (most accepts first)
- **Error** if only reject tiers exist — re-swipe or pass `--allow-reject-fallback`

Rating-mode fallback (legacy):

```bash
uv run python -m experiments.patch_sweep.record_winners \
  --phase phase1_archive_soundfonts \
  --mode rating \
  --responses .../responses/responses_....json \
  --realism-threshold 4.0
```

---

## Phase 1 (legacy) — 7 candidate banks

Quick audition of the original shortlist before the archive download:

```bash
uv run python -m experiments.patch_sweep.sweep \
  --phase phase1_soundfonts -j 8```

Listening test uses **realism only** (same as archive). To record with the old combined threshold instead:

```bash
uv run python -m experiments.patch_sweep.record_winners \
  --phase phase1_soundfonts \
  --use-mean-rating \
  --mean-rating-threshold 4.1 \
  --responses .../responses_....json
```

---

## Phase 2 — FX (on phase-1 shortlists)

Compare **3 light FX profiles** using each category's **primary** phase-1 soundfont (first in the shortlist).

### 2.1 Render

Requires `winners.yaml` phase 1 `completed: true`.

```bash
uv run python -m experiments.patch_sweep.sweep \
  --phase phase2_fx -j 8```

~3 variants × 24 stems = **72** renders.  
Output: `experiments/patch_sweep/output/phase2_fx/`

### 2.2 Build 10s clips

Same clip pipeline as phase 1 — aligned windows from basic reference, 3 clips per stem:

```bash
uv run python -m experiments.patch_sweep.make_clips \
  --sweep-dir experiments/patch_sweep/output/phase2_fx \
  --clips-per-stem 3 \
  -j 8
```

~3 FX profiles × 24 stems × 3 clips ≈ **216 swipe cards** (one category at a time).

### 2.3 Swipe listening test

```bash
uv run python -m experiments.listening.serve --sweep patch \
  --patch-sweep-dir experiments/patch_sweep/output/phase2_fx
```

Open [http://127.0.0.1:8766/swipe?type=patch](http://127.0.0.1:8766/swipe?type=patch).

Same keyboard map and checkpoint/resume behavior as phase 1. With only 3 FX variants, `?order=group_stem` is useful for direct A/B on the same 10s passage.

### 2.4 Record winners

```bash
uv run python -m experiments.patch_sweep.record_winners \
  --phase phase2_fx \
  --mode swipe \
  --responses experiments/patch_sweep/output/phase2_fx/responses/swipe_YYYYMMDDTHHMMSSZ.json
```

Uses the same tier pools as phase 1: **all** variants with ≥1 strong accept are kept; if none, up to **3** weak accepts. Legacy star-rating path remains via `--mode rating`.

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
    fx_variant_ids: [fx_dry, fx_light]
    fx_profiles: [dry, light]
    fx_profile: dry
```

Production slakh mode **randomly picks** a soundfont and FX profile per (song, category) from each shortlist, independently and deterministically from `sample_seed`. MIDI programs are unchanged (no GM pool remapping).

`synthesis/patches.py` loads this automatically as `SLAKH_CATEGORY_RENDER`.

---

## Phase 5 — Validate & run B1 ablation

### 5.1 Sanity check (probe stems)

Re-render a few probe stems in slakh mode and confirm they differ from A1:

```bash
uv run python -m experiments.patch_sweep.sweep \
  --phase phase2_fx --limit-stems 2 --limit-variants 1```

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
