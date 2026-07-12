# Rendering notes

## Output layout

Default root: `/deepfreeze/pnlong/SPDMX` (`OUTPUT_DIR` in [`shared/config.py`](../shared/config.py)).

Development artifacts live under `{OUTPUT_DIR}/dev/`. The shipped dataset is `{OUTPUT_DIR}/SPDMX/`.

**Ablation** (listening test; default `synthesize` behavior):

```
{OUTPUT_DIR}/dev/ablations/
├── basic/              # A1
├── basic_realify/      # A2
├── slakh/              # B1
└── slakh_realify/      # B2
```

**Full-scale stems** (`synthesize --full`; normally called by `build_spdmx.py`):

```
{OUTPUT_DIR}/dev/stems/           # raw synthesis
{OUTPUT_DIR}/dev/stems_realify/   # realified (optional)
```

**Analysis** (song lengths, etc.):

```
{OUTPUT_DIR}/dev/analysis/song_lengths/
```

Output symlinked in-repo at [`analysis/output/`](../analysis/output/) → `{OUTPUT_DIR}/dev/analysis/` (gitignored).

Ablation outputs symlinked at [`synthesis/ablations_output/`](../synthesis/ablations_output/) → `{OUTPUT_DIR}/dev/ablations/` (gitignored).

Create both after clone: `uv run python -m shared.setup_symlinks`

**Assembled sPDMX dataset** (via `build_spdmx.py`, not implemented yet):

```
{OUTPUT_DIR}/SPDMX/
```

## Per-song layout

```
data/<mirrored-song-path>/
├── stem_0.flac   # or stem_0.mp3 with --mp3
├── stem_1.flac
├── ...
└── mixture.flac  # or mixture.mp3 with --mp3
```

Default on-disk format is FLAC (PCM_16). Pass `--mp3` to write MP3 stems and mixtures for prototyping (less storage). Use the same `--mp3` flag for realify so it reads and writes the matching format.

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
4. Write `mixture.flac` (stem files on disk unchanged).

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

All synthesis flows go through `synthesis.synthesize`:

```bash
# A1 (default: random sample from rated_deduplicated)
python -m synthesis.synthesize --render-mode basic

# B1
python -m synthesis.synthesize --render-mode slakh

# A2 (requires A1 stems, or synthesizes first if missing)
python -m synthesis.synthesize --render-mode basic --realify

# B2
python -m synthesis.synthesize --render-mode slakh --realify

# Full PDMX after listening test
python -m synthesis.synthesize --render-mode basic --full
python -m synthesis.synthesize --render-mode basic --full --realify
```

Song-length analysis (no synthesis required):

```bash
python -m analysis.analyze_song_lengths
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

Same `ABLATION_SAMPLE_SEED` ensures basic and slakh render the same songs.

## Listening test

Subjective comparison across A1–B2 once all four dirs exist. See prior hypotheses in git history / project notes.

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
| `--full` for all valid PDMX | Done |
| `build_spdmx.py` | Stub |
| Patch pools (Slakh) | Stub |
| `mixture.flac` per song | Done |
| Listening test | Viewer available (`python -m synthesis.listening.serve`) |
| Song-length analysis (PDMX metadata + plots) | Done |
