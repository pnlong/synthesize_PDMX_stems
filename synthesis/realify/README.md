# realify

Stable Audio 3 (SA3) audio-to-audio “realification” of synthesized stems.

**Setup:** [`SETUP.md`](../../SETUP.md) at repo root — uv, submodule, flash-attention, Hugging Face.

## Files

| File | Description |
|------|-------------|
| `realify.py` | CLI and `run_realify()`: apply SA3 to each stem using captions + presets |
| `silence.py` | Post-SA3 silence enforcement (reference vs realified energy gating) |
| [`SILENCE.md`](SILENCE.md) | Paper-ready description of silence enforcement algorithm |
| `chunking.py` | Overlap-and-stitch chunking for stems longer than the model buffer |
| `presets/` | Per-category SA3 presets (`categories.yaml`) and routing rules |
| `captions/` | Generate text prompts from PDMX metadata for SA3 conditioning |
| `stable-audio-3/` | Git submodule — SA3 model code; see root [`SETUP.md`](../../SETUP.md) |
| `tests/` | Realify smoke tests (GPU) and preset exploration notebook |

## Usage

Realify reads stems from the matching non-realify ablation (`basic/` → `basic_realify/`) and **errors if that ablation has not been synthesized first**.

- **`medium`**: requires a visible GPU (`CUDA_VISIBLE_DEVICES` selects device(s); one SA3 model per GPU)
- **`small-music`**: uses GPU when visible; otherwise CPU multiprocessing with `-j`

```bash
# After basic/ exists with complete stems:
# GPU (medium); realify auto-skips GPUs without enough free VRAM
CUDA_VISIBLE_DEVICES=0,3 python -m synthesis.synthesize --render-mode basic --realify

# CPU fallback (small-music only):
python -m synthesis.synthesize --render-mode basic --realify -m small-music -j 4
```

Standalone:

```bash
python -m synthesis.realify.realify \
  --source-dir .../dev/ablations/basic \
  --output-dir .../dev/ablations/basic_realify
```

Captions are generated in memory from `data.csv` + `stems.csv` (not stored). Preview with:

```bash
python -m synthesis.realify.captions.generate --dataset_dir .../dev/ablations/basic
```

## Silence enforcement

After SA3 generation, each stem passes through [`silence.py`](silence.py) by default: overlapping windows compare reference (A1) and realified energy; hallucinated content in reference-silent regions is zeroed, with an active margin for decay tails and boundary crossfades. See [`SILENCE.md`](SILENCE.md) for the full algorithm and paper-ready writeup.

Disable for A/B tests: `--no-silence-enforce`

## Realify settings

Global constants in [`shared/config.py`](../../shared/config.py):

| Constant | Default | Purpose |
|----------|---------|---------|
| `REALIFY_INIT_NOISE_LEVEL` | `0.45` | SA3 timbre-transfer noise fallback when preset omits it |
| `REALIFY_STEPS` | `8` | Diffusion steps (rf_denoiser sweet spot) |
| `REALIFY_CFG_SCALE` | `1.0` | Classifier-free guidance scale |
| `REALIFY_SILENCE_ENFORCE` | `True` | Post-SA3 hallucination silence gating |
| `REALIFY_SILENCE_CHUNK_MS` | `500.0` | Silence analysis window length |
| `REALIFY_SILENCE_THRESHOLD_DB` | `-60.0` | Window peak below this = silent |
| `REALIFY_SILENCE_ACTIVE_MARGIN_MS` | `300.0` | Protect realified tails after reference activity |
| `REALIFY_SILENCE_FADE_MS` | `15.0` | Boundary crossfade into forced-silent regions |

[`presets/categories.yaml`](presets/categories.yaml) sets `default` plus per-category overrides (`init_noise_level`, `prompt_variant`, `steps`, `cfg_scale`). Stem `name` and `is_drum` are mapped to categories via `routing` rules.

Captions use each stem's resolved `prompt_variant`. All captions start with a fixed preservation anchor for the `current` variant, then shuffled PDMX metadata and an instrument hint.

**Long stems:** stems longer than the model buffer are split into overlapping chunks (default 2s overlap), realified separately, and crossfaded back together. Chunk size is derived from the model `sample_size` minus SA3's 6s duration padding (`~114s` for `small-music`, model-dependent for `medium`). Silence enforcement runs on the full stitched stem afterward.
