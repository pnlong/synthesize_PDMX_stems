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

Realify reads stems from the matching non-realify ablation (`basic/` → `basic_realify/`) and **errors if that ablation has not been synthesized first**.

- **`medium`**: requires a visible GPU (`CUDA_VISIBLE_DEVICES` selects device(s); one SA3 model per GPU)
- **`small-music`**: uses GPU when visible; otherwise CPU multiprocessing with `-j`

```bash
# After basic/ exists with complete stems:
CUDA_VISIBLE_DEVICES=0,1 python -m synthesis.synthesize --render-mode basic --realify

# CPU fallback (small-music only):
python -m synthesis.synthesize --render-mode basic --realify -m small-music -j 4
```

Standalone:

```bash
python -m synthesis.realify.captions.generate --dataset_dir .../dev/ablations/basic
python -m synthesis.realify.realify \
  --source-dir .../dev/ablations/basic \
  --output-dir .../dev/ablations/basic_realify
```
