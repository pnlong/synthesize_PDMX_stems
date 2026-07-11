"""Realify raw fluidsynth stems using Stable Audio 3 audio-to-audio."""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from shared.config import (
    CAPTIONS_FILE_NAME,
    DEFAULT_AUDIO_FORMAT,
    OUTPUT_DIR,
    SAMPLE_RATE,
    STEMS_FILE_NAME,
)
from synthesis.audio import (
    load_stem,
    stem_duration_seconds,
    stem_is_valid,
    stem_path,
    write_audio,
    write_mixture_from_song_dir,
)
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


def visible_cuda_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def realify_uses_gpu(model: str) -> bool:
    """Return True when realify should run on visible CUDA devices."""
    cuda_count = visible_cuda_count()
    if model == "medium":
        if cuda_count == 0:
            raise RuntimeError(
                "SA3 medium requires a GPU. Set CUDA_VISIBLE_DEVICES to select device(s), "
                "or use -m small-music for CPU realify."
            )
        return True
    # small-music: prefer GPU when visible, otherwise CPU multiprocessing
    return cuda_count > 0


def load_model(model_name: str):
    from stable_audio_3 import StableAudioModel

    return StableAudioModel.from_pretrained(model_name)


def realify_stem(
    init_audio_path: Path,
    output_path: Path,
    prompt: str,
    preset: dict,
    model,
    duration_seconds: float,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
):
    init_audio = (load_stem(init_audio_path), SAMPLE_RATE)
    audio = model.generate(
        init_audio=init_audio,
        init_noise_level=preset.get("init_noise_level", 0.65),
        prompt=prompt,
        duration=duration_seconds,
    )
    write_audio(audio, output_path, audio_format)


def resolve_stem_output_path(
    song_dir: Path,
    track: int,
    source_dir: Path,
    output_dir: Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> Path:
    out_song_dir = resolve_output_song_dir(song_dir, source_dir, output_dir)
    return stem_path(out_song_dir, track, audio_format)


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
    audio_format: str = DEFAULT_AUDIO_FORMAT,
) -> list[dict]:
    tasks = []
    for _, row in captions.iterrows():
        song_dir = Path(row["path"])
        track = int(row["track"])
        out_path = resolve_stem_output_path(
            song_dir, track, source_dir, output_dir, audio_format,
        )
        if out_path.exists():
            continue
        source_stem_path = stem_path(song_dir, track, audio_format)
        if not stem_is_valid(source_stem_path):
            continue
        tasks.append({
            "row": row.to_dict(),
            "out_path": str(out_path),
            "stem_path": str(source_stem_path),
            "duration": stem_duration_seconds(source_stem_path),
            "audio_format": audio_format,
        })
    return tasks


def _init_gpu_realify_worker(gpu_queue, model_name: str, presets_filepath: str):
    global _REALIFY_MODEL, _REALIFY_PRESETS

    import torch

    device_id = gpu_queue.get()
    torch.cuda.set_device(device_id)
    _REALIFY_PRESETS = load_presets(Path(presets_filepath))
    _REALIFY_MODEL = load_model(model_name)


def _init_cpu_realify_worker(model_name: str, presets_filepath: str):
    global _REALIFY_MODEL, _REALIFY_PRESETS

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
        audio_format=task["audio_format"],
    )
    return task["out_path"]


def _run_realify_gpu(
    tasks: list[dict],
    *,
    model: str,
    presets_filepath: Path,
) -> None:
    cuda_count = visible_cuda_count()
    n_workers = min(cuda_count, len(tasks))

    if n_workers == 1:
        import torch
        torch.cuda.set_device(0)
        sa3_model = load_model(model)
        presets = load_presets(presets_filepath)
        for task in tqdm(tasks, desc="Realifying stems (GPU)", unit="stem"):
            row = pd.Series(task["row"])
            realify_stem(
                init_audio_path=Path(task["stem_path"]),
                output_path=Path(task["out_path"]),
                prompt=row["prompt"],
                preset=select_preset(presets, row),
                model=sa3_model,
                duration_seconds=task["duration"],
                audio_format=task["audio_format"],
            )
        return

    gpu_queue = multiprocessing.Queue()
    for device_id in range(n_workers):
        gpu_queue.put(device_id)

    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_init_gpu_realify_worker,
        initargs=(gpu_queue, model, str(presets_filepath)),
    ) as pool:
        desc = f"Realifying stems ({n_workers} GPUs via CUDA_VISIBLE_DEVICES)"
        for _ in tqdm(
            pool.imap(_realify_worker, tasks, chunksize=1),
            total=len(tasks),
            desc=desc,
            unit="stem",
        ):
            pass


def _run_realify_cpu(
    tasks: list[dict],
    *,
    model: str,
    presets_filepath: Path,
    jobs: int,
) -> None:
    n_workers = min(max(jobs, 1), len(tasks))

    if n_workers == 1:
        _init_cpu_realify_worker(model, str(presets_filepath))
        for task in tqdm(tasks, desc="Realifying stems (CPU)", unit="stem"):
            _realify_worker(task)
        return

    with multiprocessing.Pool(
        processes=n_workers,
        initializer=_init_cpu_realify_worker,
        initargs=(model, str(presets_filepath)),
    ) as pool:
        for _ in tqdm(
            pool.imap(_realify_worker, tasks, chunksize=1),
            total=len(tasks),
            desc=f"Realifying stems ({n_workers} CPU workers)",
            unit="stem",
        ):
            pass


def run_realify(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str = "medium",
    limit: int | None = None,
    jobs: int = 1,
    presets_filepath: Path | None = None,
    captions_filepath: Path | None = None,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
):
    """Realify stems on visible GPU(s) or CPU (small-music only)."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    captions_filepath = captions_filepath or source_dir / f"{CAPTIONS_FILE_NAME}.csv"
    presets_filepath = presets_filepath or Path(__file__).parent / "presets.yaml"

    if output_dir != source_dir:
        copy_metadata_tables(source_dir, output_dir)

    captions = pd.read_csv(captions_filepath)
    if limit:
        captions = captions.head(limit)

    tasks = build_realify_tasks(captions, source_dir, output_dir, audio_format)
    if not tasks:
        write_mixtures_for_dataset(source_dir, output_dir, audio_format)
        return

    if realify_uses_gpu(model):
        _run_realify_gpu(tasks, model=model, presets_filepath=presets_filepath)
    else:
        _run_realify_cpu(tasks, model=model, presets_filepath=presets_filepath, jobs=jobs)

    write_mixtures_for_dataset(source_dir, output_dir, audio_format)


def resolve_output_song_dir(song_dir: Path, source_dir: Path, output_dir: Path) -> Path:
    if output_dir == source_dir:
        return song_dir
    song_dir_str = str(song_dir)
    source_prefix = str(source_dir)
    if not song_dir_str.startswith(source_prefix):
        raise ValueError(f"Song path {song_dir} is not under source dir {source_dir}")
    return Path(str(output_dir) + song_dir_str[len(source_prefix):])


def write_mixtures_for_dataset(
    source_dir: Path,
    output_dir: Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
):
    """Build mixture per song from stems in the output tree (same procedure as synthesis)."""
    stems_csv = output_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.exists():
        stems_csv = source_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.exists():
        return

    stems = pd.read_csv(stems_csv)
    for song_path, group in stems.groupby("path"):
        out_song_dir = resolve_output_song_dir(Path(song_path), source_dir, output_dir)
        tracks = sorted(int(t) for t in group["track"])
        mixture_path = write_mixture_from_song_dir(out_song_dir, tracks, audio_format)
        if mixture_path:
            print(f"Wrote mixture {mixture_path}")


def parse_args(args=None, namespace=None):
    import multiprocessing

    parser = argparse.ArgumentParser(description="Realify stems with Stable Audio 3.")
    parser.add_argument("--source-dir", default=None, type=str)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("-m", "--model", default="medium", choices=["small-music", "medium"])
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument(
        "-j",
        "--jobs",
        "--workers",
        default=int(multiprocessing.cpu_count() / 4),
        type=int,
        help="CPU worker processes for small-music realify when no GPU is visible.",
    )
    parser.add_argument(
        "--mp3",
        action="store_true",
        help="Read/write MP3 stems and mixtures instead of FLAC (must match synthesis format).",
    )
    return parser.parse_args(args=args, namespace=namespace)


def main():
    args = parse_args()
    source_dir = args.source_dir or full_stems_dir(OUTPUT_DIR)
    output_dir = args.output_dir or source_dir
    from synthesis.audio import synthesis_audio_format

    run_realify(
        source_dir,
        output_dir,
        model=args.model,
        limit=args.limit,
        jobs=args.jobs,
        audio_format=synthesis_audio_format(args.mp3),
    )


if __name__ == "__main__":
    main()
