# realify

Stable Audio 3 (SA3) audio-to-audio “realification” of synthesized stems.

**Setup:** [`SETUP.md`](../../SETUP.md) at repo root — uv, submodule, flash-attention, Hugging Face.

## Files

| File | Description |
|------|-------------|
| `realify.py` | CLI and `run_realify()`: apply SA3 to each stem using captions + presets |
| `presets.yaml` | Per-instrument SA3 generation presets (`init_noise_level`, etc.) |
| `captions/` | Generate text prompts from PDMX metadata for SA3 conditioning |
| `stable-audio-3/` | Git submodule — SA3 model code; see root [`SETUP.md`](../../SETUP.md) |
| `tests/` | Realify smoke tests (GPU) and preset exploration notebook |

## Usage

Realify is a **GPU second pass** after CPU multiprocessing synthesis. Use `--realify-gpus 0,1,...` (default: all visible GPUs) to run one SA3 model per GPU in parallel.

```bash
python -m synthesis.synthesize --render-mode basic --realify --realify-only --realify-gpus 0,1
```

Standalone:

```bash
python -m synthesis.realify.captions.generate --dataset_dir .../dev/ablations/basic
python -m synthesis.realify.realify \
  --source-dir .../dev/ablations/basic \
  --output-dir .../dev/ablations/basic_realify
```
