# Tuning methodology

How we pick production rendering settings for sPDMX ablations **A2** (basic + realify) and **B1/B2** (slakh + realify). Both sweeps share the same probe stems and listening infrastructure; they optimize different parts of the pipeline.

## Design principle: per-category winners

Neither sweep picks one global winner. Probe stems in [`probe_stems.yaml`](probe_stems.yaml) are tagged with a `category` (piano, strings, wind, drums, …). The listening test rates variants stem-by-stem; [`listening/aggregate.py`](listening/aggregate.py) reports the best variant **within each category**.

| Sweep | What varies | Locked in production |
|-------|-------------|----------------------|
| **Patch** (Slakh) | Fluidsynth soundfont, GM program pools, FX | [`synthesis/patches.py`](../synthesis/patches.py) `PATCH_POOLS` |
| **Preset** (SA3) | Phased: `init_noise_level` → prompt → diffusion | [`synthesis/realify/presets/categories.yaml`](../synthesis/realify/presets/categories.yaml) |

Categories with multiple probe stems (3 per category) vote on the same winner via mean realism/content across stems.

## Shared workflow

```mermaid
flowchart TD
    probe[probe_stems.yaml] --> sweep[Run sweep]
    sweep --> listen[listening.serve :8766]
    listen --> notes[results_notes.md]
    notes --> aggregate[listening.aggregate]
    aggregate --> lock[Lock production config]
    lock --> ablation[Full ablation A1–B2]
    ablation --> validate[synthesis.listening :8765]
```

1. **Tune** — render variants on the 24-stem probe set (3 per category; [`patch_sweep/`](patch_sweep/), [`preset_sweep/`](preset_sweep/))
2. **Evaluate** — blinded listening test ([`listening/`](listening/)); rate **content** (melody/rhythm/timing) and **realism**
3. **Aggregate** — per-category winners → `results_notes.md`
4. **Lock** — merge winners into `patches.py` or `categories.yaml`
5. **Validate** — full 100-song ablation + [`synthesis/listening`](../synthesis/listening/) comparison across A1–B2

Document decisions in each sweep’s `results_notes.md` as you go (winners, rejects, held-out checks).

### Blinded listening (all phases)

Each phase uses the same **blinded** listening test at port **8766**. You never see soundfont names, FX settings, or pool IDs during rating — only **Sample A / B / C** in randomized order per stem.

```bash
# After a phase sweep finishes:
uv run python -m experiments.listening.serve --sweep patch \
  --patch-sweep-dir experiments/patch_sweep/output/phase1_soundfonts

# Open http://127.0.0.1:8766/test?type=patch

# When done, export JSON from the UI, then:
uv run python -m experiments.listening.aggregate \
  --sweep patch \
  --responses experiments/patch_sweep/output/phase1_soundfonts/responses/responses_....json \
  --output experiments/patch_sweep/results_notes.md
```

**What you rate (per sample vs reference):**

| Scale | Question |
|-------|----------|
| Content (1–5) | Same melody, rhythm, timing as reference? |
| Realism (1–5) | Sounds like a realistic, appropriate instrument? |

**How winners are picked** (`aggregate.py`):

1. Group ratings by **category** (piano, strings, wind, …) and **variant_id**
2. Drop variants that fail a **content gate** (min stem content ≥ 3, mean ≥ 3.5) — avoids picking something that sounds “real” but changed the music
3. Among survivors, pick highest **mean realism** (tie-break: mean content), averaged across **3 probe stems per category**
4. All stems in a category vote together on one winner

Variant IDs stay in the exported JSON for aggregation only; the UI shows blind labels.

| Phase | `variant_id` encodes | Reference | Separate output dir |
|-------|---------------------|-----------|-------------------|
| 1 — Soundfonts | `sgm_v2`, `generaluser`, … | A1 basic | `output/phase1_soundfonts/` |
| 2 — FX | `fx_dry`, `fx_light`, `fx_warm` | A1 basic | `output/phase2_fx/` |
| 3 — Pools | `pool_v1_conservative`, … | A1 basic | `output/phase3_pools/` |

Phase 1 tip: content scores should be ~equal (same MIDI, different bank) — **realism** is the main signal. Phase 3: content may diverge slightly when programs change; keep the content gate.

**Not yet wired:** dedicated sweep scripts/grids for phase 1 and 2 (soundfont + FX variants in manifest). Phase 3 reuses the existing patch sweep once pools are populated.

---

## Patch sweep (Slakh / B1)

**Goal:** make `--render-mode slakh` audibly and meaningfully different from `basic`, approximating [Slakh2100](http://www.slakh.com/) stem diversity without Kontakt VSTs.

### Slakh-style variety (two layers)

1. **Per listening category** (after tuning + lock) — each category gets its own soundfont, FX profile, and program pool (`winners_locked.yaml` → `SLAKH_CATEGORY_RENDER`).
2. **Per track, within a song** — for each GM instrument class in a song (e.g. all string tracks), `select_patch()` randomly picks one GM program from that category's pool and reuses it for every track of that class **in that song**. Across songs the draw changes (seeded by `(sample_seed, song_path, gm_class)`).

Phase 3 of the patch sweep compares pool variants (`pool_v1_conservative`, `pool_v2_diverse`, `pool_v3_slakh_like`). Phase 1–2 intentionally disable pools (`pool_id: null`) so you judge soundfonts and FX on passthrough MIDI programs.

Until `winners_locked.yaml` exists, `--render-mode slakh` behaves like basic (no pool remapping). Complete phases 1–3 and run `patch_sweep.lock` to enable B1 variety.

Slakh uses 187 professional sample patches in ~34 GM classes, with per-track random assignment and baked-in reverb/EQ. Our Fluidsynth approximation:

- **Soundfont** — GM bank character (which samples Fluidsynth loads)
- **Program pools** — class-matched GM program remapping ([`patches.py`](../synthesis/patches.py))
- **FX** — light reverb/EQ (post-fluidsynth or fluidsynth flags), applied after soundfont is chosen

### Staged tuning (do not skip ahead)

| Phase | What we compare | Status |
|-------|-----------------|--------|
| **1 — Soundfonts** | 7 candidate banks, dry render, no program remap | **In progress** |
| **2 — FX** | Light reverb/EQ profiles on 1–3 shortlisted soundfonts | Pending |
| **3 — Program pools** | `pool_v1` / `pool_v2` / `pool_v3` GM program lists | Pending |
| **4 — Production lock** | Per-category winners → B1 ablation (100 songs) | Pending |

See [`patch_sweep/GUIDE.md`](patch_sweep/GUIDE.md) for the **step-by-step runbook** and [`patch_sweep/soundfonts.yaml`](patch_sweep/soundfonts.yaml) for the soundfont catalog.

### Reference anchor in listening test

Patch sweep reference = **A1 basic** synthesis (same probe stem, no slakh randomization).

---

## Preset sweep (SA3 realify / A2)

**Goal:** find SA3 audio-to-audio settings that improve timbre realism while preserving musical content, per instrument category.

### Staged tuning (do not skip ahead)

| Phase | What we compare | Status |
|-------|-----------------|--------|
| **1 — Noise** | 5 `init_noise_level` values, fixed `current` prompt | Pending |
| **2 — Prompts** | 3 prompt variants on phase-1 noise winners | Pending |
| **3 — Diffusion** | 3 steps×cfg profiles (optional) | Pending |
| **4 — Production lock** | Per-category winners → A2 ablation (100 songs) | Pending |

See [`preset_sweep/GUIDE.md`](preset_sweep/GUIDE.md) for the **step-by-step runbook** and [`preset_sweep/grids/`](preset_sweep/grids/) for phase grids.

### What we vary

| Parameter | Phase | Grid values | Fixed |
|-----------|-------|-------------|-------|
| `init_noise_level` | 1 | 0.25, 0.35, 0.45, 0.55, 0.65 | prompt=`current`, steps=8, cfg=1.0 |
| `prompt_variant` | 2 | `current`, `minimal`, `preservation` | noise from phase 1 |
| `steps` / `cfg_scale` | 3 (optional) | 3 diffusion profiles | noise + prompt from phases 1–2 |
| SA3 model | — | — | `medium` (post-trained) |

Phase 1: content scores should spread with noise — find realism peak within content gate. Phase 2: content gate strict. Phase 3: fine-tuning only if needed.

| Phase | `variant_id` encodes | Reference | Separate output dir |
|-------|---------------------|-----------|-------------------|
| 1 — Noise | `noise0.45`, … | A1 raw | `output/phase1_noise/` |
| 2 — Prompts | `current`, `minimal`, … | A1 raw | `output/phase2_prompts/` |
| 3 — Diffusion | `steps8_cfg1.0`, … | A1 raw | `output/phase3_diffusion/` |

### What we do *not* vary in this sweep

- Source stems (always A1 `basic` ablation)
- Caption generation pipeline (except prompt variant in phase 2)
- Chunking / batch size / GPU worker layout

### Reference anchor in listening test

Preset sweep reference = **A1 raw** (unrealified basic stem).

### Production target

Per-category overrides in [`synthesis/realify/presets/categories.yaml`](../synthesis/realify/presets/categories.yaml). Lock via `uv run python -m experiments.preset_sweep.lock`.

See [`preset_sweep/README.md`](preset_sweep/README.md).

---

## Probe set

[`probe_stems.yaml`](probe_stems.yaml) — **24 stems** (3 per category × 8 categories). Shared by both sweeps so patch and preset winners are comparable across categories. Each stem is validated against its MIDI `program_change` at sweep startup (e.g. voice stems must use choir/voice GM programs, not piano placeholders).

---

## Outputs and symlinks

Sweep audio lives on deepfreeze; in-repo symlinks via `uv run python -m shared.setup_symlinks`:

| Sweep | Deepfreeze path | Repo symlink |
|-------|-----------------|--------------|
| Patch | `{OUTPUT_DIR}/dev/experiments/patch_sweep/` | `experiments/patch_sweep/output/` |
| Preset | `{OUTPUT_DIR}/dev/experiments/preset_sweep/` | `experiments/preset_sweep/output/` |

Soundfont library: `soundfonts/` → `/data3/pnlong/soundfonts` (see [`patch_sweep/soundfonts.yaml`](patch_sweep/soundfonts.yaml)).

---

## Recording results

Each sweep has a `results_notes.md` template. After listening:

```bash
uv run python -m experiments.listening.aggregate \
  --sweep patch \
  --responses responses_patch.json \
  --output experiments/patch_sweep/results_notes.md
```

Fill in notes column with *why* a variant won or lost — future you (and the aggregate step) needs the reasoning, not just the scores.
