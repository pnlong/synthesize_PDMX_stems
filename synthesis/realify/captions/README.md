# captions

Generate SA3 text prompts in memory from PDMX song/stem metadata.

## Files

| File | Description |
|------|-------------|
| `generate.py` | `generate_captions()` from `data.csv` + `stems.csv`; optional CLI preview |
| `metadata.py` | `get_caption()`: build a prompt string from row metadata |
| `tests/` | Caption generation tests |

## Usage

Captions are **not written to disk**. `synthesize --realify` and `run_realify()` call `generate_captions()` in memory before the SA3 pass.

Preview prompts for a dataset:

```bash
python -m synthesis.realify.captions.generate --dataset_dir .../dev/ablations/basic
```

The listening viewer also derives captions on the fly from the same metadata tables.
