# Rendering notes

## Output layout

Default root: `/deepfreeze/pnlong/SPDMX` (`OUTPUT_DIR` in [`shared/config.py`](../shared/config.py)).

Development artifacts live under `{OUTPUT_DIR}/dev/`. The shipped dataset is `{OUTPUT_DIR}/SPDMX/`.

**Ablation** (listening test; default `synthesize` behavior):

```
{OUTPUT_DIR}/dev/ablations/
├── listening_sample.yaml   # shared stratified song/stem inventory
├── basic/                  # A1
├── basic_realify/          # A2
├── slakh/                  # B1
├── slakh_realify/          # B2
├── ddsp_basic/             # CA1 (neural DDSP + basic soundfont fallback copies)
├── ddsp_basic_realify/     # CA2
├── ddsp_slakh/             # CB1 (neural DDSP + slakh soundfont fallback copies)
├── ddsp_slakh_realify/     # CB2
└── clips/{condition}/      # aligned 10s MP3 clips for listening.serve
```

If an older `slakh_ddsp/` tree exists, rename it to `ddsp_slakh/` (and `*_realify` likewise).

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
uv run python -m analysis.prepare_synthesis --subset all_valid -j 8
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
└── ...           # mix = sum(stems); no mixture.* on disk
```

Default on-disk format is **MP3**. Pass `--flac` to write FLAC stems (PCM_16). Use the same `--flac` flag for realify / mix so they read and write the matching format.

## Mixture procedure

Canonical description (equations, motivation, constants): **[`MIXING.md`](MIXING.md)**.

Constant across all ablations (A1–B2), basic and slakh, synthesis and realify:

| Setting | Value |
|---|---|
| Sample rate | 44.1 kHz |
| Stem channels | `STEM_CHANNELS` in `shared/config.py` (default `1` mono; `2` keeps fluidsynth/SA3 stereo) |
| Loudness | Applied in `synthesis.mix` (−23 LUFS BS.1770-4, peak-limited to 1.0) |

1. Load raw stems; loudness-normalize toward −23 LUFS (BS.1770) with per-stem peak limiting at 1.0, then pad to equal length.
2. Multiply each stem by MIDI velocity dynamics \(s_i = v_i^{\max} / v_{\mathrm{song}}^{\max}\) (note-ons with velocity \(> 0\)).
3. Sum stems sample-wise (in memory).
4. If mixture peak > `MIXTURE_PEAK_LIMIT` (1.0), apply uniform gain `limit / peak` to every stem (same factor), so released stems remain linearly summable.
5. Overwrite `stem_*.mp3` / `stem_*.flac` with the scaled waveforms. **No `mixture.*` is written by default** — the mix is just `sum(stems)` (`--write-mixture` to also write it).

**Synthesis and realify write raw stems** (no LUFS). Summability normalization is a separate pass:

```bash
uv run python -m synthesis.mix --stems-dir /path/to/ablation -j 8
# or:
uv run python -m synthesis.mix --render-mode basic -j 8
uv run python -m synthesis.mix --render-mode basic --realify -j 8
# Preview without overwrite + write mixtures:
uv run python -m synthesis.mix --render-mode basic --no-overwrite --write-mixture -j 8
```

Implemented in [`audio.py`](audio.py), [`velocity.py`](velocity.py), [`mix.py`](mix.py). `synthesize` / `realify` print the suggested mix command when they finish.

## Two-pass pipeline (synthesis + realify)

Synthesis and realify are intentionally separate passes with different hardware profiles:

| Pass | Work | Parallelism | Hardware |
|------|------|-------------|----------|
| 1 — Synthesis | Fluidsynth render (basic or slakh) | `-j` / `--jobs` multiprocessing pool | CPU |
| 2 — Realify | SA3 audio-to-audio per stem | One process per visible GPU; stems sorted category→length; batch size auto from per-GPU VRAM (`REALIFY_BATCH_SIZE=0`, or `--realify-batch-size N`) | GPU / CPU |

Pass 1 writes raw stems under `dev/ablations/{basic,slakh}/` or `dev/stems/`. Pass 2 reads those stems, runs captions + SA3, and writes to `{mode}_realify/` (or `stems_realify/`). **Pass 2 never re-synthesizes** — it errors if the raw ablation is incomplete. Mixtures are a separate `synthesis.mix` pass afterward.

Use `CUDA_VISIBLE_DEVICES` to select GPU(s). `medium` requires a visible GPU. `small-music` uses GPU when available, otherwise CPU multiprocessing with `-j`.

```bash
# Prerequisite — GM register (once; re-run after alias YAML edits)
python -m analysis.prepare_synthesis --subset all_valid -j 8

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
COMMON="--sample-seed 43 -j 8"

# Step 0
python -m analysis.prepare_synthesis --subset all_valid -j 8

# Donors (stratified sample written on first run → listening_sample.yaml)
python -m synthesis.synthesize --render-mode basic $COMMON          # A1
python -m synthesis.synthesize --render-mode slakh $COMMON          # B1
python -m synthesis.synthesize --render-mode basic --realify $COMMON  # A2
python -m synthesis.synthesize --render-mode slakh --realify $COMMON  # B2

# DDSP (copies soundfont fallbacks from donors; renders neural stems only)
python -m synthesis.synthesize --render-mode ddsp_basic $COMMON          # CA1
python -m synthesis.synthesize --render-mode ddsp_slakh $COMMON          # CB1
python -m synthesis.synthesize --render-mode ddsp_basic --realify $COMMON  # CA2
python -m synthesis.synthesize --render-mode ddsp_slakh --realify $COMMON  # CB2

# Optional: peak-normalize stems so mix = sum(stems)
python -m synthesis.mix --render-mode basic -j 8
python -m synthesis.mix --render-mode basic --realify -j 8

# Aligned 10s clips (windows from A1) + listening viewer
python -m synthesis.listening.make_clips --clip-seconds 10
python -m synthesis.listening.serve

# Full PDMX after listening test (dense corrected MIDI)
python -m analysis.prepare_synthesis --subset all_valid -j 8
python -m synthesis.synthesize --render-mode basic --full
python -m synthesis.synthesize --render-mode basic --full --realify
python -m synthesis.mix --full -j 8
```

Synthesize always uses dense corrected MIDIs from `dev/mid_corrected/` (`prepare_synthesis` is the step-0 setup).

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
| CA1 | `ddsp_basic` | `dev/ablations/ddsp_basic/` |
| CA2 | `ddsp_basic`, `--realify` | `dev/ablations/ddsp_basic_realify/` |
| CB1 | `ddsp_slakh` | `dev/ablations/ddsp_slakh/` |
| CB2 | `ddsp_slakh`, `--realify` | `dev/ablations/ddsp_slakh_realify/` |

Shared stratified sample (`listening_sample.yaml`, seed 43, ≥50 stems/category) ensures all eight conditions render the same songs.

**Donor reuse (NFS-safe copies):** CA/CB soundfont-fallback stems are `copy2`'d from A/B (raw) and A2/B2 (realify). Neural stems are newly rendered / SA3'd. Provenance is in `ddsp_routing.csv`:

| Column | Meaning |
|--------|---------|
| `path` | Song directory in this ablation (same as `stems.csv`) |
| `source` | `rendered` or `reused:basic` / `reused:slakh` |
| `original_path` | Absolute stem filepath copied from; `NA` when newly rendered |

No symlinks/hardlinks.

### Slakh mode (`--render-mode slakh`)

Slakh-style rendering adds **per-track patch variety** on top of basic Fluidsynth:

- Each listening category (piano, strings, wind, …) can use a different soundfont, FX profile, and GM program pool (from patch sweep tuning → `winners_locked.yaml`).
- Within a song, each track randomly draws a program from its category's pool (`select_patch` in [`patches.py`](patches.py)). Tracks sharing the same GM instrument class in a song get the **same** patch; the draw varies across songs (seeded by `(sample_seed, song_path, gm_class)`).
- Pools are defined in `PATCH_POOLS` (`pool_v1_conservative`, `pool_v2_diverse`, `pool_v3_slakh_like`). Until winners are locked, slakh mode passes MIDI programs through unchanged (same as basic).

See [`experiments/TUNING.md`](../experiments/TUNING.md) for the phased tuning workflow (soundfonts → FX → pools).

### Neural DDSP modes (`ddsp_basic` / `ddsp_slakh`)

Hybrid per-stem backends. Soundfont fallbacks copy from **basic** (`ddsp_basic`) or **slakh** (`ddsp_slakh`):

| Stem | Backend |
|------|---------|
| Acoustic hammer piano (GM **0, 1, 3**) / acoustic piano names | **DDSP-Piano** (MAESTRO; polyphony OK) |
| Harpsichord, clavinet, e-piano, electric grand (GM 2, 4–7) | donor soundfont copy |
| 13 URMP instruments, **monophonic** | **MIDI-DDSP** |
| Timbre mismatches (piccolo, pan flute, english horn, muted trumpet, …) | donor soundfont copy |
| Polyphonic URMP-eligible stems | donor soundfont copy |
| Drums, guitar, bass guitar, vocals, synths, other | donor soundfont copy |

Routing details live in [`synthesis/ddsp/routing.py`](ddsp/routing.py) (`DDSP_PIANO_PROGRAMS`, name deny-lists, SATB-vs-sax vocal guard).

- CA2/CB2 SA3 only neural stems; fallback stems copy from A2/B2.
- Neural models run in an isolated TF venv (`.venv-ddsp`); see SETUP Track C. Linux x86_64 only.
- **Persistent multi-GPU pool** (default): one long-lived `worker serve` process per id in `CUDA_VISIBLE_DEVICES`. Synthesis runs **three global passes** — (1) all `ddsp_piano` stems, (2) all `midi_ddsp` stems, (3) donor/soundfont — restarting the pool between neural passes so only one TF model is hot. Within a pass, same-backend stems in a song fan out across GPUs. Song-level jobs stay at `-j 1`. `SPDMX_DDSP_ONESHOT=1` = legacy per-stem subprocesses; `SPDMX_DDSP_FORCE_CPU=1` → CPU worker.
- Routing decisions are written to `ddsp_routing.csv` beside the ablation tables.
- Provenance: [`THIRD_PARTY.md`](../THIRD_PARTY.md). Vocals deliberately stay on soundfont(+SA3); lyric SVS is out of scope.

## Listening test

Stem-level comparison across A1–CB2. After ablations, build aligned **10s** clips (windows chosen from A1) and serve the clips tree:

```bash
uv run python -m synthesis.listening.make_clips --clip-seconds 10
uv run python -m synthesis.listening.serve
```

See [`listening/README.md`](listening/README.md).

## Status

| Feature | Status |
|---------|--------|
| Mono + BS.1770 stems | Done |
| `--render-mode` + `--realify` on synthesize | Done |
| `--render-mode ddsp_basic` / `ddsp_slakh` + donor copy reuse | Done (isolated TF venv; SETUP Track C) |
| Stratified listening sample + 10s clips | Done |
| `--full` for all valid PDMX | Done |
| `build_spdmx.py` | Stub |
| Patch pools (Slakh) | Stub |
| `mixture` per song | Not stored by default; `synthesis.mix` applies LUFS × velocity × peak so mix = sum(stems). See [`MIXING.md`](MIXING.md). |
| Listening test | Viewer available (`python -m synthesis.listening.serve`) |
| Song-length analysis (PDMX metadata + plots) | Done |
| Neural-DDSP coverage (`analysis.ddsp_coverage`) | Done |
