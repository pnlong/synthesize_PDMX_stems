# captions

Generate SA3 text prompts from PDMX song/stem metadata.

## Files

| File | Description |
|------|-------------|
| `generate.py` | CLI: write `captions.csv` for a synthesized dataset directory |
| `metadata.py` | `get_caption()`: sample a prompt string from row metadata (genres, artist, etc.) |
| `tests/` | Caption generation tests |

## Output

Writes `captions.csv` alongside `data.csv` and `stems.csv` in the dataset dir:

| Column | Description |
|--------|-------------|
| `path` | Song output directory |
| `track` | Stem index |
| `prompt` | Text prompt for SA3 |

Called automatically by `synthesize --realify` before the realify pass.
