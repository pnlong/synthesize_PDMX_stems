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

Output symlinked in-repo at [`analysis/output/`](../analysis/output/) → `{OUTPUT_DIR}/dev/analysis/`.

**Assembled sPDMX dataset** (via `build_spdmx.py`, not implemented yet):

```
{OUTPUT_DIR}/SPDMX/
```

## Per-song layout

```
data/<mirrored-song-path>/
├── stem_0.flac
├── stem_1.flac
├── ...
└── mixture.flac
```

## Mixture procedure

Constant across all ablations (A1–B2), basic and slakh, synthesis and realify:

1. Stems are BS.1770-normalized to −23 LUFS and padded to equal length.
2. Sum stems sample-wise.
3. If mixture peak > `MIXTURE_PEAK_LIMIT` (1.0), apply uniform gain `limit / peak`.
4. Write `mixture.flac` (stem files on disk unchanged).

Implemented in [`audio.py`](audio.py). Called from `synthesize.py` after stems and from `realify.py` after realify completes.

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

Standalone realify/captions (optional):

```bash
python -m synthesis.realify.captions.generate --dataset_dir .../dev/ablations/basic
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

## Status

| Feature | Status |
|---------|--------|
| Mono + BS.1770 stems | Done |
| `--render-mode` + `--realify` on synthesize | Done |
| `--full` for all valid PDMX | Done |
| `build_spdmx.py` | Stub |
| Patch pools (Slakh) | Stub |
| `mixture.flac` per song | Done |
| Listening test | Not started |
| Song-length analysis (PDMX metadata + plots) | Done |
