"""Realify raw fluidsynth stems using Stable Audio 3 audio-to-audio."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml

from shared.config import CAPTIONS_FILE_NAME, OUTPUT_DIR, STEMS_FILE_NAME
from synthesis.audio import write_mixture_from_song_dir
from synthesis.paths import full_stems_dir


def load_presets(presets_filepath: Path) -> dict:
    with open(presets_filepath) as f:
        return yaml.safe_load(f)


def select_preset(presets: dict, row: pd.Series) -> dict:
    if row.get("is_drum"):
        return presets.get("drums", presets["default"])
    name = row.get("name")
    if isinstance(name, str) and name.lower() in presets:
        return presets[name.lower()]
    return presets["default"]


def realify_stem(
    init_audio_path: Path,
    output_path: Path,
    prompt: str,
    preset: dict,
    model_name: str,
    duration_seconds: float,
):
    import torchaudio
    from stable_audio_3 import StableAudioModel

    model = StableAudioModel.from_pretrained(model_name)
    init_audio = torchaudio.load(str(init_audio_path))
    audio = model.generate(
        init_audio=init_audio,
        init_noise_level=preset.get("init_noise_level", 0.65),
        prompt=prompt,
        duration=duration_seconds,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), audio, sample_rate=44100)


def copy_metadata_tables(source_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("data.csv", "stems.csv", f"{CAPTIONS_FILE_NAME}.csv"):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def run_realify(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str = "medium",
    limit: int | None = None,
    presets_filepath: Path | None = None,
    captions_filepath: Path | None = None,
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    captions_filepath = captions_filepath or source_dir / f"{CAPTIONS_FILE_NAME}.csv"
    presets_filepath = presets_filepath or Path(__file__).parent / "presets.yaml"

    if output_dir != source_dir:
        copy_metadata_tables(source_dir, output_dir)

    captions = pd.read_csv(captions_filepath)
    presets = load_presets(presets_filepath)
    if limit:
        captions = captions.head(limit)

    for _, row in captions.iterrows():
        song_dir = Path(row["path"])
        track = int(row["track"])
        stem_path = song_dir / f"stem_{track}.flac"

        if output_dir == source_dir:
            out_path = song_dir / f"stem_{track}.flac"
        else:
            song_dir_str = str(song_dir)
            source_prefix = str(source_dir)
            if not song_dir_str.startswith(source_prefix):
                raise ValueError(f"Song path {song_dir} is not under source dir {source_dir}")
            out_path = Path(str(output_dir) + song_dir_str[len(source_prefix):]) / f"stem_{track}.flac"

        if out_path.exists():
            continue

        preset = select_preset(presets, row)
        import soundfile as sf
        duration = sf.info(str(stem_path)).frames / sf.info(str(stem_path)).samplerate

        realify_stem(
            init_audio_path=stem_path,
            output_path=out_path,
            prompt=row["prompt"],
            preset=preset,
            model_name=model,
            duration_seconds=duration,
        )
        print(f"Realified {out_path}")

    write_mixtures_for_dataset(source_dir, output_dir)


def resolve_output_song_dir(song_dir: Path, source_dir: Path, output_dir: Path) -> Path:
    if output_dir == source_dir:
        return song_dir
    song_dir_str = str(song_dir)
    source_prefix = str(source_dir)
    if not song_dir_str.startswith(source_prefix):
        raise ValueError(f"Song path {song_dir} is not under source dir {source_dir}")
    return Path(str(output_dir) + song_dir_str[len(source_prefix):])


def write_mixtures_for_dataset(source_dir: Path, output_dir: Path):
    """Build mixture.flac per song from stems in the output tree (same procedure as synthesis)."""
    stems_csv = output_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.exists():
        stems_csv = source_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.exists():
        return

    stems = pd.read_csv(stems_csv)
    for song_path, group in stems.groupby("path"):
        out_song_dir = resolve_output_song_dir(Path(song_path), source_dir, output_dir)
        tracks = sorted(int(t) for t in group["track"])
        mixture_path = write_mixture_from_song_dir(out_song_dir, tracks)
        if mixture_path:
            print(f"Wrote mixture {mixture_path}")


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description="Realify stems with Stable Audio 3.")
    parser.add_argument("--source-dir", default=None, type=str)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("-m", "--model", default="medium", choices=["small-music", "medium"])
    parser.add_argument("--limit", default=None, type=int)
    return parser.parse_args(args=args, namespace=namespace)


def main():
    args = parse_args()
    source_dir = args.source_dir or full_stems_dir(OUTPUT_DIR)
    output_dir = args.output_dir or source_dir
    run_realify(source_dir, output_dir, model=args.model, limit=args.limit)


if __name__ == "__main__":
    main()
