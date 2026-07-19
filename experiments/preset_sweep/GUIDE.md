# SA3 preset tuning — step-by-step runbook

Follow these steps in order. Each phase has a **blinded listening test** (you only see Sample A/B/C, not noise levels or prompt names). Winners chain into the next phase.

**Prerequisites**

```bash
# A1 basic ablation probe stems exist
ls synthesis/ablations_output/basic/data/

# GPU with enough VRAM for SA3 medium (see synthesis/realify/README.md)
# Optional: smaller MP3 outputs for listening
export SWEEP_MP3="--mp3"
```

---

## Phase 1 — init_noise_level (fixed prompt)

Compare **5 noise levels** with `prompt_variant: current`, `steps: 8`, `cfg_scale: 1.0`. Pick the best **per category** (piano, strings, wind, …).

### 1.1 Render

```bash
uv run python -m experiments.preset_sweep.sweep \
  --phase phase1_noise $SWEEP_MP3
```

~5 variants × 24 stems = **120** SA3 forwards.  
Output: `experiments/preset_sweep/output/phase1_noise/`

### 1.2 Blinded listening test

```bash
uv run python -m experiments.listening.serve --sweep preset \
  --preset-sweep-dir experiments/preset_sweep/output/phase1_noise
```

Open [http://127.0.0.1:8766/test?type=preset](http://127.0.0.1:8766/test?type=preset)

- Reference = A1 raw (unrealified basic stem)
- Rate each blinded sample: **content** + **realism**
- Phase 1 tip: content scores should spread with noise level — winners need **mean content ≥ 4.5**, then highest **realism**; ties on realism → **higher noise**
- Export JSON when all stems are complete (Save to server or download)

### 1.3 Record winners

```bash
uv run python -m experiments.preset_sweep.record_winners \
  --phase phase1_noise \
  --responses experiments/preset_sweep/output/phase1_noise/responses/responses_YYYYMMDDTHHMMSSZ.json
```

This writes per-category `variant_id` (e.g. `noise0.45`, `noise0.55`) into [`winners.yaml`](winners.yaml). Phase 1 uses mean content ≥ **4.5**, then highest realism; ties on realism go to the **higher noise level** (override with `--noise-content-threshold`).

Optional: inspect aggregate report:

```bash
uv run python -m experiments.listening.aggregate \
  --sweep preset \
  --sweep-dir experiments/preset_sweep/output/phase1_noise \
  --responses experiments/preset_sweep/output/phase1_noise/responses/responses_....json \
  --output experiments/preset_sweep/results_notes.md
```

---

## Phase 1b — Noise audit on diverse clips (before prompts)

After phase 1, stress-test each category's winning noise level on **diverse 10-second clips** from the full A1 ablation (not the fixed probe set). Clips are auto-trimmed to 10s and silent stems are skipped.

**Production silence enforcement is applied during render** (same post-SA3 pass as A2 realify). Rest hallucinations are corrected before you listen, so the audit focuses on **timbre and active-region content**, not invented noise in rests.

For each category you blind-compare **phase-1 winner** vs **one-step-lower** noise. Pick by **realism** among variants that pass the content gate (played sections preserve melody/rhythm/timing). If the lower noise wins on realism, `record_winners` **lowers phase-1** automatically.

### 1b.1 Render

Requires `winners.yaml` phase 1 `completed: true`.

```bash
uv run python -m experiments.preset_sweep.sweep \
  --phase phase1b_noise_audit $SWEEP_MP3
```

Builds `clips/` (10s references) + realified variants with silence enforcement enabled. Default: **5 diverse stems per category** (40 total) × ~2 noise levels each.

Output: `experiments/preset_sweep/output/phase1b_noise_audit/`

### 1b.2 Listen → record

```bash
uv run python -m experiments.listening.serve --sweep preset \
  --preset-sweep-dir experiments/preset_sweep/output/phase1b_noise_audit

uv run python -m experiments.preset_sweep.record_winners \
  --phase phase1b_noise_audit \
  --responses experiments/preset_sweep/output/phase1b_noise_audit/responses/responses_....json
```

Phase 1b tip: rate **content** for **played sections only** — rests are silence-corrected. Use **realism** as the main differentiator between winner and lower-noise variants. If the lower-noise sample sounds more realistic and passes content, it wins and phase-1 noise is revised downward.

---

## Phase 2 — prompt_variant (on phase-1 winners)

Compare **3 prompt formats** using each category's winning noise level from phase 1 (after phase 1b audit).

### 2.1 Render

Requires `winners.yaml` phases 1 and 1b `completed: true`.

```bash
uv run python -m experiments.preset_sweep.sweep \
  --phase phase2_prompts $SWEEP_MP3
```

~3 variants × 24 stems = **72** SA3 forwards.  
Output: `experiments/preset_sweep/output/phase2_prompts/`

### 2.2 Listen → record

```bash
uv run python -m experiments.listening.serve --sweep preset \
  --preset-sweep-dir experiments/preset_sweep/output/phase2_prompts

uv run python -m experiments.preset_sweep.record_winners \
  --phase phase2_prompts \
  --responses experiments/preset_sweep/output/phase2_prompts/responses/responses_....json
```

Phase 2 tip: apply the **content gate** strictly — pick highest realism among variants that preserve melody/rhythm/timing.

---

## Phase 3 — Diffusion budget (optional)

Compare **3 steps × cfg_scale profiles** using each category's locked noise + prompt from phases 1–2.

Skip this phase if phases 1–2 already sound good; lock will use default `steps: 8`, `cfg_scale: 1.0`.

### 3.1 Render

Requires phase 1 and 2 winners.

```bash
uv run python -m experiments.preset_sweep.sweep \
  --phase phase3_diffusion $SWEEP_MP3
```

~3 variants × 24 stems = **72** SA3 forwards.  
Output: `experiments/preset_sweep/output/phase3_diffusion/`

### 3.2 Listen → record

```bash
uv run python -m experiments.listening.serve --sweep preset \
  --preset-sweep-dir experiments/preset_sweep/output/phase3_diffusion

uv run python -m experiments.preset_sweep.record_winners \
  --phase phase3_diffusion \
  --responses experiments/preset_sweep/output/phase3_diffusion/responses/responses_....json
```

---

## Phase 4 — Verification render (diverse 1b corpus)

Render **locked presets** on the same **diverse 10s clips** from phase 1b (~40 stems). Requires phases 1, 1b, and 2 complete in [`winners.yaml`](winners.yaml).

```bash
uv run python -m experiments.preset_sweep.sweep \
  --phase phase4_verify_diverse $SWEEP_MP3
```

Reuses `phase1b_noise_audit/clips/` (symlinked). One realified variant per stem (`locked`).  
Output: `experiments/preset_sweep/output/phase4_verify_diverse/`

To force re-render after winner changes:

```bash
rm -rf experiments/preset_sweep/output/phase4_verify_diverse/variants
```

---

## Phase 5 — Final verification (listen + bypass)

Compare **basic clip** (reference) vs **locked-preset realified** audio on the diverse corpus. No blind-test responses file is needed.

```bash
uv run python -m experiments.listening.serve --sweep preset
```

Open [http://127.0.0.1:8766/verify?type=preset](http://127.0.0.1:8766/verify?type=preset).

- **Bypass all stems in this category** — shortcut identical to checking bypass on every stem below
- **Bypass realification for this instrument** — per-stem; partial bypass becomes routing rules at lock (track name keywords, or GM program if unnamed)
- When every stem in a category is bypassed, lock writes `categories.<name>.realify: false`

**Finish** writes `verification_final_winners.yaml_YYYYMMDDTHHMMSSZ.json` under `phase4_verify_diverse/responses/`.

---

## Phase 6 — Lock production config

When verification looks good:

```bash
uv run python -m experiments.preset_sweep.lock \
  --verification experiments/preset_sweep/output/phase4_verify_diverse/responses/verification_final_winners.yaml_....json
```

(`--verification` is optional — omit to lock auto-winners as-is.)

Writes:

- [`winners_locked.yaml`](winners_locked.yaml) — audit trail of per-category presets
- [`synthesis/realify/presets/categories.yaml`](../../synthesis/realify/presets/categories.yaml) — production overrides merged into `categories:`

Example locked entry:

```yaml
categories:
  piano:
    init_noise_level: 0.45
    prompt_variant: minimal
    steps: 8
    cfg_scale: 1.0
  organ:
    realify: false
```

To preview without touching `categories.yaml`:

```bash
uv run python -m experiments.preset_sweep.lock --skip-categories-yaml
```

---

## Phase 7 — Validate & run A2 ablation

### 7.1 Sanity check (probe stems)

Re-listen on 2–3 held-out stems not in the probe set, or spot-check a few probe stems with the locked config.

### 7.2 Full A2 ablation

```bash
uv run python -m synthesis.synthesize --render-mode basic --realify
```

Output: `dev/ablations/basic_realify/`

### 7.3 Ablation listening comparison

```bash
uv run python -m synthesis.listening.serve
```

Compare A1 (basic) vs A2 (basic_realify) on port **8765**.

---

---

## Quick reference

| Step | Command |
|------|---------|
| Render phase N | `uv run python -m experiments.preset_sweep.sweep --phase <phase>` |
| Listen (per phase) | `uv run python -m experiments.listening.serve --sweep preset --preset-sweep-dir <phase_dir>` |
| Final verify | Render phase 4 first, then `http://127.0.0.1:8766/verify?type=preset` |
| Record winners | `uv run python -m experiments.preset_sweep.record_winners --phase <phase> --responses <json>` |
| Lock production | `uv run python -m experiments.preset_sweep.lock [--verification <json>]` |
| A2 ablation | `uv run python -m synthesis.synthesize --render-mode basic --realify` |

| Phase | `--phase` value | Variants | Needs prior winners | Silence enforce |
|-------|-----------------|----------|---------------------|-----------------|
| 1 Noise | `phase1_noise` | 5 | — | no |
| 1b Audit | `phase1b_noise_audit` | ~2 per category | phase 1 | **yes** |
| 2 Prompts | `phase2_prompts` | 3 | phase 1 + 1b | no |
| 3 Diffusion | `phase3_diffusion` | 3 | phase 1 + 2 (optional before lock) | no |
| 4 Verify render | `phase4_verify_diverse` | 1 (`locked`) | phase 1 + 1b + 2 | no |

## Files

| File | Purpose |
|------|---------|
| [`grids/`](grids/) | Variant definitions per phase |
| [`winners.yaml`](winners.yaml) | Your per-phase decisions (updated by `record_winners`) |
| [`winners_locked.yaml`](winners_locked.yaml) | Locked production config (written by `lock`) |
| [`preset_grid.yaml`](preset_grid.yaml) | Optional flat factorial for validation |
| [`results_notes.md`](results_notes.md) | Subjective notes template |

## Troubleshooting

- **Phase 2/3 sweep refuses to start** — run `record_winners` for prior phases first; phase 2 needs phase **1b** audit complete.
- **Missing probe stem** — ensure A1 basic ablation exists for all `probe_stems.yaml` song IDs.
- **GPU OOM** — reduce `--realify-batch-size` or run one stem at a time with `--limit-stems 1`.
- **Listening UI shows "Audio not available"** — sweep didn't finish; check manifest.csv `out_path` files exist.
