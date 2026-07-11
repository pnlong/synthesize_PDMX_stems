# realify

Stable Audio 3 (SA3) audio-to-audio “realification” of synthesized stems.

**Setup:** [`SETUP.md`](SETUP.md) — uv, submodule, flash-attention, Hugging Face.

## Files

| File | Description |
|------|-------------|
| `realify.py` | CLI and `run_realify()`: apply SA3 to each stem using captions + presets |
| `presets.yaml` | Per-instrument SA3 generation presets (`init_noise_level`, etc.) |
| `captions/` | Generate text prompts from PDMX metadata for SA3 conditioning |
| `stable-audio-3/` | Git submodule — SA3 model code; setup in [`SETUP.md`](SETUP.md) |
| `tests/` | Realify smoke tests (GPU) and preset exploration notebook |

## Usage

Normally invoked via `python -m synthesis.synthesize --realify`. Standalone:

```bash
python -m synthesis.realify.captions.generate --dataset_dir .../dev/ablations/basic
python -m synthesis.realify.realify \
  --source-dir .../dev/ablations/basic \
  --output-dir .../dev/ablations/basic_realify
```
