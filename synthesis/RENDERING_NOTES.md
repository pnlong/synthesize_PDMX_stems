# Rendering notes

## Output layout

Default root: `/deepfreeze/pnlong/SPDMX` (`OUTPUT_DIR` in [`shared/config.py`](../shared/config.py)).

Development artifacts live under `{OUTPUT_DIR}/dev/`. The shipped dataset is `{OUTPUT_DIR}/SPDMX/`.

**Ablation** (listening test; default `synthesize` behavior):

```
{OUTPUT_DIR}/dev/ablations/
├── basic/                  # A1
├── basic_realify/          # A2
├── slakh/                  # B1
├── slakh_realify/          # B2
├── slakh_ddsp/             # B3 (MIDI-DDSP + DDSP-Piano + slakh fallback)
└── slakh_ddsp_realify/     # B4 (optional SA3 on B3 stems)
```

**Full-scale stems** (`synthesize --full`; normally called by `build_spdmx.py`):

```
{OUTPUT_DIR}/dev/stems/           # raw synthesis
{OUTPUT_DIR}/dev/stems_realify/   # realified (optional)
```

**Analysis** (song lengths, GM register, etc.):

```
{OUTPUT_DIR}/dev/analysis/song_lengths/
{OUTPUT_DIR}/dev/analysis/instruments/all_valid/   # register.csv (step 0)
```

Output symlinked in-repo at [`analysis/output/`](../analysis/output/) → `{OUTPUT_DIR}/dev/analysis/` (gitignored).

Ablation outputs symlinked at [`synthesis/ablations_output/`](../synthesis/ablations_output/) → `{OUTPUT_DIR}/dev/ablations/` (gitignored).

Create both after clone: `uv run python -m shared.setup_symlinks`

**Assembled sPDMX dataset** (via `build_spdmx.py`, not implemented yet):

```
{OUTPUT_DIR}/SPDMX/
```

## GM register (step 0)

Before any ablation or `--full` run, build the per-track GM correction table:

```bash
uv run python -m analysis.analyze_gm_register --subset all_valid -j 8
```

- **Aliases:** [`analysis/gm_register_aliases.yaml`](../analysis/gm_register_aliases.yaml) (SATB→choir, harpsichord, sax vs alto, …)
- **Output:** `{OUTPUT_DIR}/dev/analysis/instruments/all_valid/register.csv` (+ corrections CSV, summary JSON, top-corrections CSV)
- **Re-run** after editing the alias YAML; then re-synthesize affected ablations with `--reset` if needed
- **Synthesize** loads that path by default (`--register` / `--no-register` to override)

Then run A1/B1/… as usual.

## Per-song layout

```
data/<mirrored-song-path>/
├── stem_0.mp3    # or stem_0.flac with --flac
├── stem_1.mp3
├── ...
└── mixture.mp3   # or mixture.flac with --flac
```

Default on-disk format is **MP3**. Pass `--flac` to write FLAC stems and mixtures (PCM_16). Use the same `--flac` flag for realify so it reads and writes the matching format.

## Mixture procedure

Constant across all ablations (A1–B2), basic and slakh, synthesis and realify:

| Setting | Value |
|---|---|
| Sample rate | 44.1 kHz |
| Stem channels | `STEM_CHANNELS` in `shared/config.py` (default `1` mono; `2` keeps fluidsynth/SA3 stereo) |
| Loudness | −23 LUFS integrated (BS.1770-4), peak-limited to 1.0 |

1. Stems are loudness-normalized toward −23 LUFS (BS.1770) with per-stem peak limiting at 1.0, then padded to equal length.
2. Sum stems sample-wise.
3. If mixture peak > `MIXTURE_PEAK_LIMIT` (1.0), apply uniform gain `limit / peak`.
4. Write `mixture.mp3` (or `mixture.flac` with `--flac`; stem files on disk unchanged).

Implemented in [`audio.py`](audio.py). Called from `synthesize.py` after stems and from `realify.py` after realify completes.

## Two-pass pipeline (synthesis + realify)

Synthesis and realify are intentionally separate passes with different hardware profiles:

| Pass | Work | Parallelism | Hardware |
|------|------|-------------|----------|
| 1 — Synthesis | Fluidsynth render (basic or slakh) | `-j` / `--jobs` multiprocessing pool | CPU |
| 2 — Realify | SA3 audio-to-audio per stem | One process per visible GPU; `--realify-batch-size` batches stems per forward pass | GPU / CPU |

Pass 1 writes raw stems under `dev/ablations/{basic,slakh}/` or `dev/stems/`. Pass 2 reads those stems, runs captions + SA3, and writes to `{mode}_realify/` (or `stems_realify/`). **Pass 2 never re-synthesizes** — it errors if the raw ablation is incomplete. Mixture rebuild at the end uses `-j` / `--jobs` CPU workers (same flag as synthesis).

Use `CUDA_VISIBLE_DEVICES` to select GPU(s). `medium` requires a visible GPU. `small-music` uses GPU when available, otherwise CPU multiprocessing with `-j`.

```bash
# Prerequisite — GM register (once; re-run after alias YAML edits)
python -m analysis.analyze_gm_register --subset all_valid -j 8

# Pass 1 — CPU multiprocessing (required first)
python -m synthesis.synthesize --render-mode basic -j 8

# Pass 2 — GPU (medium); limit devices with CUDA_VISIBLE_DEVICES
# Realify skips GPUs with <10 GiB free (see REALIFY_MIN_GPU_FREE_GB in shared/config.py).
# On mixed 3090/2080 Ti boxes, prefer the larger cards:
CUDA_VISIBLE_DEVICES=0,3 python -m synthesis.synthesize --render-mode basic --realify

# Pass 2 — CPU smoke test (small-music, no GPU)
python -m synthesis.synthesize --render-mode basic --realify -m small-music -j 4
```

Standalone realify after pass 1 (captions generated in memory):

```bash
python -m synthesis.realify.realify \
  --source-dir .../dev/ablations/basic \
  --output-dir .../dev/ablations/basic_realify
```

## Commands

All synthesis flows go through `synthesis.synthesize` (expects GM `register.csv` unless `--no-register`):

```bash
# Step 0
python -m analysis.analyze_gm_register --subset all_valid -j 8

# A1 (default: random sample from rated_deduplicated)
python -m synthesis.synthesize --render-mode basic

# B1
python -m synthesis.synthesize --render-mode slakh

# A2 (requires A1 stems, or synthesizes first if missing)
python -m synthesis.synthesize --render-mode basic --realify

# B2
python -m synthesis.synthesize --render-mode slakh --realify

# B3 — neural DDSP (MIDI-DDSP + DDSP-Piano) on slakh base; see SETUP Track C
python -m synthesis.synthesize --render-mode slakh_ddsp

# B4 — optional SA3 after B3
python -m synthesis.synthesize --render-mode slakh_ddsp --realify

# Full PDMX after listening test
python -m synthesis.synthesize --render-mode basic --full
python -m synthesis.synthesize --render-mode basic --full --realify
```

Song-length analysis (no synthesis required):

```bash
python -m analysis.analyze_song_lengths
```

Neural-DDSP coverage (for the paper / sampling design):

```bash
python -m analysis.ddsp_coverage --subset rated_deduplicated
python -m analysis.ddsp_coverage --subset rated_deduplicated --check-monophony -n 500
```

### Assembled dataset (`build_spdmx.py`, stub)

```bash
python -m synthesis.build_spdmx --render-mode basic
```

Standalone realify (captions generated in memory):

```bash
python -m synthesis.realify.realify --source-dir .../dev/ablations/basic --output-dir .../dev/ablations/basic_realify
```

## Module layout

```
synthesis/
├── synthesize.py       # main CLI (--render-mode, --full, --realify)
├── build_spdmx.py      # assemble {OUTPUT_DIR}/SPDMX/ (stub)
├── realify/
│   ├── realify.py      # SA3 audio-to-audio
│   ├── captions/       # caption generation
│   └── stable-audio-3/ # git submodule
```

## Ablation study

| ID | Flags | Output |
|----|-------|--------|
| A1 | `basic` | `dev/ablations/basic/` |
| A2 | `basic`, `--realify` | `dev/ablations/basic_realify/` |
| B1 | `slakh` | `dev/ablations/slakh/` |
| B2 | `slakh`, `--realify` | `dev/ablations/slakh_realify/` |
| B3 | `slakh_ddsp` | `dev/ablations/slakh_ddsp/` |
| B4 | `slakh_ddsp`, `--realify` | `dev/ablations/slakh_ddsp_realify/` |

Same `ABLATION_SAMPLE_SEED` ensures basic / slakh / slakh_ddsp render the same songs.

### Slakh mode (`--render-mode slakh`)

Slakh-style rendering adds **per-track patch variety** on top of basic Fluidsynth:

- Each listening category (piano, strings, wind, …) can use a different soundfont, FX profile, and GM program pool (from patch sweep tuning → `winners_locked.yaml`).
- Within a song, each track randomly draws a program from its category's pool (`select_patch` in [`patches.py`](patches.py)). Tracks sharing the same GM instrument class in a song get the **same** patch; the draw varies across songs (seeded by `(sample_seed, song_path, gm_class)`).
- Pools are defined in `PATCH_POOLS` (`pool_v1_conservative`, `pool_v2_diverse`, `pool_v3_slakh_like`). Until winners are locked, slakh mode passes MIDI programs through unchanged (same as basic).

See [`experiments/TUNING.md`](../experiments/TUNING.md) for the phased tuning workflow (soundfonts → FX → pools).

### Neural DDSP mode (`--render-mode slakh_ddsp`, B3)

Hybrid per-stem backends on the **slakh** soundfont base:

| Stem | Backend |
|------|---------|
| GM piano (0–7) / piano track names | **DDSP-Piano** (MAESTRO; polyphony OK) |
| 13 URMP instruments, **monophonic** | **MIDI-DDSP** |
| Polyphonic URMP-eligible stems | slakh soundfont fallback |
| Drums, guitar, bass guitar, vocals, synths, other | slakh soundfont fallback |

- Default **no SA3** on neural stems (B3). Optional B4 runs the existing realify pass on completed B3 stems.
- Neural models run in an isolated TF venv (`.venv-ddsp`); see SETUP Track C. Linux x86_64 only.
- **GPU by default** (CUDA 12 / cuDNN 8 pip wheels + `LD_LIBRARY_PATH`); override with `SPDMX_DDSP_CUDA_VISIBLE_DEVICES` or `SPDMX_DDSP_FORCE_CPU=1`. Use `-j 1` (flock-serialized).
- Routing decisions are written to `ddsp_routing.csv` beside the ablation tables.
- Provenance: [`THIRD_PARTY.md`](../THIRD_PARTY.md). Vocals deliberately stay on soundfont(+SA3); lyric SVS is out of scope.

**Listening protocol (recommended):**

1. **Isolated stems** — same notes under B1 vs B2 vs B3 for piano and one MIDI-DDSP instrument (cleanest signal).
2. **Full-mix** — prefer pieces with high neural coverage (piano + strings/winds), not random draws dominated by drums/guitar.
3. Report corpus coverage via `python -m analysis.ddsp_coverage`.

## Listening test

Subjective comparison across A1–B4 once dirs exist. See prior hypotheses in git history / project notes.

Browse and compare generated audio locally:

```bash
uv run python -m synthesis.listening.serve
```

See [`listening/README.md`](listening/README.md).

## Status

| Feature | Status |
|---------|--------|
| Mono + BS.1770 stems | Done |
| `--render-mode` + `--realify` on synthesize | Done |
| `--render-mode slakh_ddsp` (B3 neural DDSP) | Done (isolated TF venv; SETUP Track C) |
| `--full` for all valid PDMX | Done |
| `build_spdmx.py` | Stub |
| Patch pools (Slakh) | Stub |
| `mixture.flac` per song | Done |
| Listening test | Viewer available (`python -m synthesis.listening.serve`) |
| Song-length analysis (PDMX metadata + plots) | Done |
| Neural-DDSP coverage (`analysis.ddsp_coverage`) | Done |
