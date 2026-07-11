# synthesis

Turn PDMX symbolic MIDI into mono FLAC stems; optionally realify with Stable Audio 3.

See also [`RENDERING_NOTES.md`](RENDERING_NOTES.md) for ablation design, Slakh alignment, and output layout.

## Entry points

| Script | Purpose |
|--------|---------|
| `synthesize.py` | Main CLI: ablation sample (default) or `--full` PDMX, `--render-mode`, `--realify` |
| `build_spdmx.py` | Assemble final dataset at `{OUTPUT_DIR}/SPDMX/` (stub) |

## Source files

| File | Description |
|------|-------------|
| `synthesize.py` | MIDI → fluidsynth → mono FLAC stems; optional SA3 realify pass |
| `build_spdmx.py` | Planned: copy PDMX metadata + call `synthesize --full` → `{OUTPUT_DIR}/SPDMX/` |
| `audio.py` | fluidsynth rendering, mono downmix, BS.1770-4 loudness, FLAC I/O, mixture build |
| `dataset.py` | Ablation sampling vs full-dataset filtering |
| `paths.py` | Output path helpers (`dev/ablations/`, `dev/stems/`, `dev/analysis/`, `SPDMX/`) |
| `patches.py` | Slakh-style patch randomization (stub) |
| `cli_common.py` | Shared argparse flags for synthesis CLIs |
| `realify/` | Stable Audio 3 audio-to-audio wrapper and captions |
| `tests/` | Synthesis unit tests |

## Output layout (development)

```
{OUTPUT_DIR}/dev/
├── ablations/{basic,basic_realify,slakh,slakh_realify}/
├── stems/              # synthesize --full
└── stems_realify/
```

Final dataset: `{OUTPUT_DIR}/SPDMX/` (via `build_spdmx`).
