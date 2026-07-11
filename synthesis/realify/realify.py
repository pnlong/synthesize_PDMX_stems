"""Realify raw fluidsynth stems using Stable Audio 3 audio-to-audio."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

from shared.config import CAPTIONS_FILE_NAME, OUTPUT_DIR, STEMS_FILE_NAME
from synthesis.audio import write_mixture_from_song_dir
from synthesis.paths import full_stems_dir

_REALIFY_MODEL = None
_REALIFY_PRESETS: dict | None = None


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


def parse_gpu_ids(gpus: str | None) -> list[int]:
    import torch

    if gpus is None:
        count = torch.cuda.device_count()
        if count == 0:
            raise RuntimeError("No CUDA GPUs available for realify.")
        return list(range(count))
    if gpus.strip().lower() == "all":
        count = torch.cuda.device_count()
        if count == 0:
            raise RuntimeError("No CUDA GPUs available for realify.")
        return list(range(count))
    return [int(part.strip()) for part in gpus.split(",") if part.strip()]


def load_model(model_name: str, gpu_id: int | None = None):
    if gpu_id is not None:
        import torch
        torch.cuda.set_device(gpu_id)
    from stable_audio_3 import StableAudioModel

    return StableAudioModel.from_pretrained(model_name)


def realify_stem(
    init_audio_path: Path,
    output_path: Path,
    prompt: str,
    preset: dict,
    model,
    duration_seconds: float,
):
    import torchaudio

    init_audio = torchaudio.load(str(init_audio_path))
    audio = model.generate(
        init_audio=init_audio,
        init_noise_level=preset.get("init_noise_level", 0.65),
        prompt=prompt,
        duration=duration_seconds,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), audio, sample_rate=44100)


def resolve_stem_output_path(
    song_dir: Path,
    track: int,
    source_dir: Path,
    output_dir: Path,
) -> Path:
    if output_dir == source_dir:
        return song_dir / f"stem_{track}.flac"
    song_dir_str = str(song_dir)
    source_prefix = str(source_dir)
    if not song_dir_str.startswith(source_prefix):
        raise ValueError(f"Song path {song_dir} is not under source dir {source_dir}")
    return Path(str(output_dir) + song_dir_str[len(source_prefix):]) / f"stem_{track}.flac"


def copy_metadata_tables(source_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("data.csv", "stems.csv", f"{CAPTIONS_FILE_NAME}.csv"):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def build_realify_tasks(
    captions: pd.DataFrame,
    source_dir: Path,
    output_dir: Path,
) -> list[dict]:
    tasks = []
    for _, row in captions.iterrows():
        song_dir = Path(row["path"])
        track = int(row["track"])
        out_path = resolve_stem_output_path(song_dir, track, source_dir, output_dir)
        if out_path.exists():
            continue
        stem_path = song_dir / f"stem_{track}.flac"
        info = sf.info(str(stem_path))
        tasks.append({
            "row": row.to_dict(),
            "out_path": str(out_path),
            "stem_path": str(stem_path),
            "duration": info.frames / info.samplerate,
        })
    return tasks


def _init_realify_worker(gpu_queue, model_name: str, presets_filepath: str):
    global _REALIFY_MODEL, _REALIFY_PRESETS

    gpu_id = gpu_queue.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    _REALIFY_PRESETS = load_presets(Path(presets_filepath))
    _REALIFY_MODEL = load_model(model_name)


def _realify_worker(task: dict) -> str:
    row = pd.Series(task["row"])
    preset = select_preset(_REALIFY_PRESETS, row)
    realify_stem(
        init_audio_path=Path(task["stem_path"]),
        output_path=Path(task["out_path"]),
        prompt=row["prompt"],
        preset=preset,
        model=_REALIFY_MODEL,
        duration_seconds=task["duration"],
    )
    return task["out_path"]


def run_realify(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str = "medium",
    limit: int | None = None,
    gpus: str | None = None,
    presets_filepath: Path | None = None,
    captions_filepath: Path | None = None,
):
    """GPU pass: realify stems with one SA3 model per GPU process."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    captions_filepath = captions_filepath or source_dir / f"{CAPTIONS_FILE_NAME}.csv"
    presets_filepath = presets_filepath or Path(__file__).parent / "presets.yaml"

    if output_dir != source_dir:
        copy_metadata_tables(source_dir, output_dir)

    captions = pd.read_csv(captions_filepath)
    if limit:
        captions = captions.head(limit)

    tasks = build_realify_tasks(captions, source_dir, output_dir)
    if not tasks:
        write_mixtures_for_dataset(source_dir, output_dir)
        return

    gpu_ids = parse_gpu_ids(gpus)
    n_workers = min(len(gpu_ids), len(tasks))

    if n_workers == 1:
        sa3_model = load_model(model, gpu_id=gpu_ids[0])
        presets = load_presets(presets_filepath)
        for task in tqdm(tasks, desc=f"Realifying stems (GPU {gpu_ids[0]})", unit="stem"):
            row = pd.Series(task["row"])
            realify_stem(
                init_audio_path=Path(task["stem_path"]),
                output_path=Path(task["out_path"]),
                prompt=row["prompt"],
                preset=select_preset(presets, row),
                model=sa3_model,
                duration_seconds=task["duration"],
            )
    else:
        gpu_queue = multiprocessing.Queue()
        for gpu_id in gpu_ids[:n_workers]:
            gpu_queue.put(gpu_id)

        with multiprocessing.Pool(
            processes=n_workers,
            initializer=_init_realify_worker,
            initargs=(gpu_queue, model, str(presets_filepath)),
        ) as pool:
            desc = f"Realifying stems ({n_workers} GPUs: {','.join(str(g) for g in gpu_ids[:n_workers])})"
            for _ in tqdm(
                pool.imap(_realify_worker, tasks, chunksize=1),
                total=len(tasks),
                desc=desc,
                unit="stem",
            ):
                pass

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
    parser.add_argument(
        "--gpus",
        default=None,
        type=str,
        help="Comma-separated GPU ids (default: all visible GPUs).",
    )
    return parser.parse_args(args=args, namespace=namespace)


def main():
    args = parse_args()
    source_dir = args.source_dir or full_stems_dir(OUTPUT_DIR)
    output_dir = args.output_dir or source_dir
    run_realify(source_dir, output_dir, model=args.model, limit=args.limit, gpus=args.gpus)


if __name__ == "__main__":
    main()
