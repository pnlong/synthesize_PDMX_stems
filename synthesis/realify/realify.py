"""Realify raw fluidsynth stems using Stable Audio 3 audio-to-audio."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import shutil
import warnings
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from shared.config import (
    ABLATION_SAMPLE_SEED,
    DEFAULT_AUDIO_FORMAT,
    OUTPUT_DIR,
    REALIFY_BATCH_SIZE,
    REALIFY_CFG_SCALE,
    REALIFY_CHUNKED_DECODE,
    REALIFY_INIT_NOISE_LEVEL,
    REALIFY_MIN_GPU_FREE_GB,
    REALIFY_STEPS,
    SAMPLE_RATE,
    STEMS_FILE_NAME,
)
from synthesis.audio import (
    ensure_stem_channels,
    load_stem,
    stem_duration_seconds,
    stem_is_valid,
    stem_path,
    write_audio,
    write_mixture_from_song_dir,
)
from synthesis.paths import full_stems_dir
from synthesis.realify.captions.generate import generate_captions
from synthesis.realify.chunking import (
    max_realify_chunk_samples,
    needs_chunking,
    plan_chunk_spans,
    realify_overlap_samples,
    stitch_chunk_outputs,
)
from synthesis.realify.preset_config import (
    DEFAULT_PRESETS_FILE,
    load_presets,
    preset_key,
    select_preset,
)

_REALIFY_MODEL = None
_REALIFY_PRESETS: dict | None = None
_REALIFY_BATCH_SIZE = REALIFY_BATCH_SIZE
_REALIFY_WORKER_CONFIG: dict | None = None


def stem_seed(sample_seed: int, song_path: str, track: int) -> int:
    return (sample_seed + hash((song_path, track))) % (2**31)


def build_generate_kwargs(
    *,
    preset: dict,
    model,
    prompt: str | list[str],
    duration_seconds: float | list[float],
    init_audio,
    seed: int | list[int],
    batch_size: int = 1,
) -> dict:
    kwargs = {
        "init_audio": init_audio,
        "init_noise_level": preset.get("init_noise_level", REALIFY_INIT_NOISE_LEVEL),
        "prompt": prompt,
        "duration": duration_seconds,
        "steps": preset.get("steps", REALIFY_STEPS),
        "cfg_scale": preset.get("cfg_scale", REALIFY_CFG_SCALE),
        "seed": seed,
        "batch_size": batch_size,
        "sample_size": model.model_config["sample_size"],
        "disable_tqdm": True,
        "chunked_decode": REALIFY_CHUNKED_DECODE,
    }
    negative_prompt = preset.get("negative_prompt")
    if negative_prompt is not None:
        kwargs["negative_prompt"] = negative_prompt
    return kwargs


def task_preset(task: dict, presets: dict) -> dict:
    if "preset" in task:
        return dict(task["preset"])
    row = pd.Series(task["row"])
    return select_preset(presets, row)


def task_needs_chunking(task: dict, model) -> bool:
    return needs_chunking(int(task["duration"] * SAMPLE_RATE), model)


def iter_realify_batches(
    tasks: list[dict],
    model,
    presets: dict,
    batch_size: int,
):
    """Yield task groups that can share one SA3 forward pass."""
    if batch_size <= 1:
        for task in tasks:
            yield [task]
        return

    buffer: list[dict] = []
    buffer_key: tuple | None = None

    def flush():
        nonlocal buffer, buffer_key
        if buffer:
            yield_now = buffer
            buffer = []
            buffer_key = None
            return yield_now
        return None

    for task in tasks:
        preset = task_preset(task, presets)
        key = preset_key(preset)
        if task_needs_chunking(task, model):
            pending = flush()
            if pending is not None:
                yield pending
            yield [task]
            continue

        if buffer and (len(buffer) >= batch_size or key != buffer_key):
            yield buffer
            buffer = []
            buffer_key = None

        buffer.append(task)
        buffer_key = key

    if buffer:
        yield buffer


def visible_cuda_count() -> int:
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def gpu_memory_snapshot(device_index: int) -> tuple[float, float, str]:
    """Return (free_gb, total_gb, device_name) for a visible CUDA device."""
    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return (
        free_bytes / (1024**3),
        total_bytes / (1024**3),
        torch.cuda.get_device_name(device_index),
    )


def select_realify_gpu_indices(
    *,
    min_free_gb: float = REALIFY_MIN_GPU_FREE_GB,
    log_skips: bool = True,
) -> list[int]:
    """Pick visible GPUs with enough free VRAM for an SA3 worker."""
    count = visible_cuda_count()
    if count == 0:
        return []

    selected: list[int] = []
    for device_index in range(count):
        free_gb, total_gb, name = gpu_memory_snapshot(device_index)
        if free_gb >= min_free_gb:
            selected.append(device_index)
            continue
        if log_skips:
            print(
                f"Realify: skipping GPU {device_index} ({name}) — "
                f"{free_gb:.1f} GiB free of {total_gb:.1f} GiB "
                f"(need {min_free_gb:.1f} GiB free per worker)"
            )
    return selected


def describe_visible_gpus(*, min_free_gb: float | None = None) -> str:
    count = visible_cuda_count()
    if count == 0:
        return "no CUDA devices visible"
    try:
        import torch

        parts = []
        for device_index in range(count):
            free_gb, total_gb, name = gpu_memory_snapshot(device_index)
            parts.append(f"{name} ({free_gb:.1f}/{total_gb:.1f} GiB free)")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
        summary = f"{count} GPU(s): {', '.join(parts)} (CUDA_VISIBLE_DEVICES={visible})"
        if min_free_gb is not None:
            usable = select_realify_gpu_indices(min_free_gb=min_free_gb, log_skips=False)
            summary += f"; {len(usable)} usable at >= {min_free_gb:.1f} GiB free"
        return summary
    except Exception:
        return f"{count} GPU(s)"


def reset_realify_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def log_realify_plan(
    *,
    source_dir: Path,
    output_dir: Path,
    model: str,
    n_tasks: int,
    n_captions: int,
    use_gpu: bool,
    batch_size: int = REALIFY_BATCH_SIZE,
) -> None:
    backend = "GPU" if use_gpu else "CPU"
    print(f"Realify source: {source_dir}")
    print(f"Realify output: {output_dir}")
    print(f"Realify model: {model} ({backend})")
    if batch_size > 1:
        print(f"Realify batch size: {batch_size} stems per forward pass")
    if use_gpu:
        print(
            f"Realify devices: {describe_visible_gpus(min_free_gb=REALIFY_MIN_GPU_FREE_GB)}"
        )
    else:
        print("Realify devices: CPU workers (no visible CUDA GPU)")
    print(f"Realify stems queued: {n_tasks} of {n_captions}")
    if n_tasks == 0:
        print(
            "Realify: all stems already exist in the output tree; "
            "skipping SA3 and rebuilding mixtures only."
        )


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


def should_use_flash_attention(device_index: int | None = None) -> bool:
    """FlashAttention requires Ampere (sm_80) or newer."""
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(device_index)
    return major >= 8


def _patch_sa3_disable_flash_attention() -> None:
    import stable_audio_3.models.transformer as transformer_mod

    transformer_mod.flash_attn_func = None
    transformer_mod.flash_attn_kvpacked_func = None
    transformer_mod.flash_attn_varlen_func = None
    # Avoid torch.compile flex_attention attempts on the SDPA fallback path.
    transformer_mod.flex_attention_compiled = None


def configure_sa3_env() -> None:
    """Set process env before PyTorch import to keep SA3 output quiet."""
    os.environ.setdefault("TORCH_LOGS", "-dynamo,-inductor,-dynamic")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _silence_sa3_loggers() -> None:
    for logger_name in (
        "torch",
        "torch._dynamo",
        "torch._inductor",
        "torch.fx",
        "transformers",
        "huggingface_hub",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def configure_sa3_runtime(*, device_index: int | None = None) -> None:
    """Silence noisy SA3 load warnings and disable FlashAttention when unsupported."""
    configure_sa3_env()
    _silence_sa3_loggers()
    warnings.filterwarnings(
        "ignore",
        message=r".*weight_norm.*",
        category=FutureWarning,
    )
    warnings.filterwarnings("ignore", module=r"torch\._dynamo.*")
    if should_use_flash_attention(device_index):
        return

    _patch_sa3_disable_flash_attention()
    try:
        import torch
    except ImportError:
        print("Realify: CUDA unavailable; using SA3 SDPA attention fallback.")
        return

    if device_index is None:
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
        else:
            print("Realify: CUDA unavailable; using SA3 SDPA attention fallback.")
            return

    major, minor = torch.cuda.get_device_capability(device_index)
    print(
        f"Realify: GPU {device_index} (sm {major}.{minor}) does not support "
        "FlashAttention; using SA3 SDPA attention fallback."
    )


def load_model(model_name: str, device_index: int | None = None):
    configure_sa3_runtime(device_index=device_index)
    from stable_audio_3 import StableAudioModel
    import torch

    if device_index is not None:
        device = f"cuda:{device_index}"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return StableAudioModel.from_pretrained(model_name, device=device)


def _normalize_generated_audio(audio) -> torch.Tensor:
    """Return stem-format float32 tensor (channels, samples) on CPU."""
    if hasattr(audio, "ndim") and audio.ndim == 3:
        audio = audio[0]
    return ensure_stem_channels(audio)


def _generate_realify_audio(
    *,
    preset: dict,
    model,
    prompt: str | list[str],
    duration_seconds: float | list[float],
    init_audio,
    seed: int | list[int],
    batch_size: int = 1,
) -> torch.Tensor:
    kwargs = build_generate_kwargs(
        preset=preset,
        model=model,
        prompt=prompt,
        duration_seconds=duration_seconds,
        init_audio=init_audio,
        seed=seed,
        batch_size=batch_size,
    )
    try:
        audio = model.generate(**kwargs)
    except torch.cuda.OutOfMemoryError:
        import torch

        torch.cuda.empty_cache()
        kwargs["chunked_decode"] = True
        audio = model.generate(**kwargs)
    return audio


def _normalize_generated_batch(audio) -> list[torch.Tensor]:
    if hasattr(audio, "ndim") and audio.ndim == 3:
        return [_normalize_generated_audio(audio[i : i + 1]) for i in range(audio.shape[0])]
    return [_normalize_generated_audio(audio)]


def realify_stem(
    init_audio_path: Path,
    output_path: Path,
    prompt: str,
    preset: dict,
    model,
    duration_seconds: float,
    seed: int,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
):
    waveform = load_stem(init_audio_path)
    total_samples = waveform.shape[-1]
    chunk_samples = max_realify_chunk_samples(model)
    overlap_samples = realify_overlap_samples()

    if not needs_chunking(total_samples, model):
        init_audio = (SAMPLE_RATE, waveform)
        audio = _generate_realify_audio(
            preset=preset,
            model=model,
            prompt=prompt,
            init_audio=init_audio,
            duration_seconds=duration_seconds,
            seed=seed,
        )
        write_audio(_normalize_generated_audio(audio), output_path, audio_format)
        return

    spans = plan_chunk_spans(total_samples, chunk_samples, overlap_samples)
    chunks = []
    for chunk_index, (start, end) in enumerate(spans):
        chunk_waveform = waveform[..., start:end]
        chunk_duration = (end - start) / SAMPLE_RATE
        chunk_seed = (seed + chunk_index) % (2**31)
        chunks.append(
            _normalize_generated_audio(
                _generate_realify_audio(
                    preset=preset,
                    model=model,
                    prompt=prompt,
                    init_audio=(SAMPLE_RATE, chunk_waveform),
                    duration_seconds=chunk_duration,
                    seed=chunk_seed,
                )
            )
        )

    stitched = stitch_chunk_outputs(chunks, spans, overlap_samples)
    write_audio(stitched, output_path, audio_format)


def realify_stems_batch(
    tasks: list[dict],
    *,
    model,
    presets: dict,
    audio_format: str,
) -> None:
    """Realify multiple stems in one SA3 forward pass."""
    if len(tasks) == 1:
        task = tasks[0]
        row = pd.Series(task["row"])
        realify_stem(
            init_audio_path=Path(task["stem_path"]),
            output_path=Path(task["out_path"]),
            prompt=row["prompt"],
            preset=task_preset(task, presets),
            model=model,
            duration_seconds=task["duration"],
            seed=task["seed"],
            audio_format=audio_format,
        )
        return

    rows = [pd.Series(task["row"]) for task in tasks]
    preset = task_preset(tasks[0], presets)
    waveforms = [load_stem(Path(task["stem_path"])) for task in tasks]
    audio = _generate_realify_audio(
        preset=preset,
        model=model,
        prompt=[row["prompt"] for row in rows],
        duration_seconds=[task["duration"] for task in tasks],
        init_audio=[(SAMPLE_RATE, waveform) for waveform in waveforms],
        seed=[task["seed"] for task in tasks],
        batch_size=len(tasks),
    )
    for task, stem_audio in zip(tasks, _normalize_generated_batch(audio)):
        write_audio(stem_audio, Path(task["out_path"]), audio_format)


def process_realify_tasks(
    tasks: list[dict],
    *,
    model,
    presets: dict,
    audio_format: str,
    batch_size: int,
    desc: str,
    show_progress: bool = True,
) -> None:
    batches = list(iter_realify_batches(tasks, model, presets, batch_size))
    progress = tqdm(total=len(tasks), desc=desc, unit="stem", disable=not show_progress)
    try:
        for batch in batches:
            realify_stems_batch(
                batch,
                model=model,
                presets=presets,
                audio_format=audio_format,
            )
            progress.update(len(batch))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        progress.close()


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
    for name in ("data.csv", "stems.csv"):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def build_realify_tasks(
    captions: pd.DataFrame,
    source_dir: Path,
    output_dir: Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    sample_seed: int = ABLATION_SAMPLE_SEED,
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
            "seed": stem_seed(sample_seed, str(song_dir), track),
        })
    return tasks


def _shutdown_pool(pool) -> None:
    """Gracefully shut down a multiprocessing pool.

    Avoid ``with pool`` — Pool.__exit__ calls terminate(), which force-kills CUDA
    workers and triggers resource_tracker semaphore leak warnings on shutdown.
    """
    pool.close()
    pool.join()


def _teardown_gpu_realify_worker() -> None:
    global _REALIFY_MODEL, _REALIFY_PRESETS

    _REALIFY_MODEL = None
    _REALIFY_PRESETS = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _init_gpu_realify_worker(model_name: str, presets_filepath: str, batch_size: int):
    global _REALIFY_MODEL, _REALIFY_PRESETS, _REALIFY_BATCH_SIZE, _REALIFY_WORKER_CONFIG

    configure_sa3_env()
    _REALIFY_WORKER_CONFIG = {
        "model_name": model_name,
        "presets_filepath": presets_filepath,
        "batch_size": batch_size,
    }
    _REALIFY_BATCH_SIZE = batch_size
    _REALIFY_MODEL = None
    _REALIFY_PRESETS = None


def _ensure_gpu_realify_worker(device_id: int) -> None:
    global _REALIFY_MODEL, _REALIFY_PRESETS, _REALIFY_BATCH_SIZE

    if _REALIFY_MODEL is not None:
        return
    if _REALIFY_WORKER_CONFIG is None:
        raise RuntimeError("GPU realify worker config was not initialized")

    import torch

    cfg = _REALIFY_WORKER_CONFIG
    torch.cuda.set_device(device_id)
    _REALIFY_BATCH_SIZE = cfg["batch_size"]
    _REALIFY_PRESETS = load_presets(Path(cfg["presets_filepath"]))
    _REALIFY_MODEL = load_model(cfg["model_name"], device_index=device_id)


def _init_cpu_realify_worker(model_name: str, presets_filepath: str, batch_size: int):
    global _REALIFY_MODEL, _REALIFY_PRESETS, _REALIFY_BATCH_SIZE

    configure_sa3_env()
    _REALIFY_BATCH_SIZE = batch_size
    _REALIFY_PRESETS = load_presets(Path(presets_filepath))
    _REALIFY_MODEL = load_model(model_name, device_index=None)


def _realify_gpu_worker_shard(args: tuple[int, list[dict]]) -> int:
    device_id, shard = args
    if not shard:
        return 0
    try:
        _ensure_gpu_realify_worker(device_id)
        process_realify_tasks(
            shard,
            model=_REALIFY_MODEL,
            presets=_REALIFY_PRESETS,
            audio_format=shard[0]["audio_format"],
            batch_size=_REALIFY_BATCH_SIZE,
            desc="Realifying stems",
            show_progress=False,
        )
        return len(shard)
    finally:
        _teardown_gpu_realify_worker()


def _realify_worker_shard(shard: list[dict]) -> int:
    if not shard:
        return 0
    process_realify_tasks(
        shard,
        model=_REALIFY_MODEL,
        presets=_REALIFY_PRESETS,
        audio_format=shard[0]["audio_format"],
        batch_size=_REALIFY_BATCH_SIZE,
        desc="Realifying stems",
        show_progress=False,
    )
    return len(shard)


def _run_realify_gpu(
    tasks: list[dict],
    *,
    model: str,
    presets_filepath: Path,
    batch_size: int,
    audio_format: str,
) -> None:
    gpu_indices = select_realify_gpu_indices()
    if not gpu_indices:
        raise RuntimeError(
            "No GPU has enough free VRAM for realify. "
            f"Need at least {REALIFY_MIN_GPU_FREE_GB:.1f} GiB free per worker. "
            f"Visible devices: {describe_visible_gpus()}. "
            "Free memory on busy GPUs, set CUDA_VISIBLE_DEVICES to idle devices "
            "(e.g. CUDA_VISIBLE_DEVICES=0,3), or use -m small-music for CPU realify."
        )

    n_workers = min(len(gpu_indices), len(tasks))

    if n_workers == 1:
        configure_sa3_env()
        import torch

        device_index = gpu_indices[0]
        torch.cuda.set_device(device_index)
        free_gb, total_gb, name = gpu_memory_snapshot(device_index)
        print(
            f"Realify: loading SA3 {model} on GPU {device_index} "
            f"({name}, {free_gb:.1f}/{total_gb:.1f} GiB free)"
        )
        sa3_model = load_model(model, device_index=device_index)
        presets = load_presets(presets_filepath)
        process_realify_tasks(
            tasks,
            model=sa3_model,
            presets=presets,
            audio_format=audio_format,
            batch_size=batch_size,
            desc="Realifying stems (GPU)",
        )
        return

    device_labels = []
    for device_id in gpu_indices[:n_workers]:
        free_gb, total_gb, name = gpu_memory_snapshot(device_id)
        device_labels.append(
            f"GPU {device_id} ({name}, {free_gb:.1f}/{total_gb:.1f} GiB free)"
        )
    print(
        f"Realify: loading SA3 {model} on {n_workers} GPU worker(s): "
        + "; ".join(device_labels)
    )

    ctx = multiprocessing.get_context("spawn")
    shards = [tasks[i::n_workers] for i in range(n_workers)]
    shard_args = [
        (gpu_indices[i], shards[i])
        for i in range(n_workers)
    ]
    pool = ctx.Pool(
        processes=n_workers,
        initializer=_init_gpu_realify_worker,
        initargs=(model, str(presets_filepath), batch_size),
    )
    try:
        desc = f"Realifying stems ({n_workers} GPUs)"
        with tqdm(total=len(tasks), desc=desc, unit="stem") as progress:
            for count in pool.imap(_realify_gpu_worker_shard, shard_args, chunksize=1):
                progress.update(count)
    finally:
        _shutdown_pool(pool)


def _run_realify_cpu(
    tasks: list[dict],
    *,
    model: str,
    presets_filepath: Path,
    jobs: int,
    batch_size: int,
    audio_format: str,
) -> None:
    n_workers = min(max(jobs, 1), len(tasks))

    if n_workers == 1:
        _init_cpu_realify_worker(model, str(presets_filepath), batch_size)
        process_realify_tasks(
            tasks,
            model=_REALIFY_MODEL,
            presets=_REALIFY_PRESETS,
            audio_format=audio_format,
            batch_size=batch_size,
            desc="Realifying stems (CPU)",
        )
        return

    shards = [tasks[i::n_workers] for i in range(n_workers)]
    pool = multiprocessing.Pool(
        processes=n_workers,
        initializer=_init_cpu_realify_worker,
        initargs=(model, str(presets_filepath), batch_size),
    )
    try:
        with tqdm(
            total=len(tasks),
            desc=f"Realifying stems ({n_workers} CPU workers)",
            unit="stem",
        ) as progress:
            for count in pool.imap(_realify_worker_shard, shards, chunksize=1):
                progress.update(count)
    finally:
        _shutdown_pool(pool)


def run_realify(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    model: str = "medium",
    limit: int | None = None,
    jobs: int = 1,
    batch_size: int = REALIFY_BATCH_SIZE,
    presets_filepath: Path | None = None,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    sample_seed: int = ABLATION_SAMPLE_SEED,
    reset: bool = False,
):
    """Realify stems on visible GPU(s) or CPU (small-music only)."""
    configure_sa3_env()
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    presets_filepath = presets_filepath or DEFAULT_PRESETS_FILE

    if reset:
        reset_realify_output(output_dir)

    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if output_dir != source_dir:
        copy_metadata_tables(source_dir, output_dir)

    presets = load_presets(presets_filepath)
    captions = generate_captions(source_dir, seed=sample_seed, presets=presets)
    if limit:
        captions = captions.head(limit)

    tasks = build_realify_tasks(
        captions, source_dir, output_dir, audio_format, sample_seed=sample_seed,
    )
    use_gpu = realify_uses_gpu(model) if tasks else False
    log_realify_plan(
        source_dir=source_dir,
        output_dir=output_dir,
        model=model,
        n_tasks=len(tasks),
        n_captions=len(captions),
        use_gpu=use_gpu,
        batch_size=batch_size,
    )
    if not tasks:
        write_mixtures_for_dataset(
            source_dir, output_dir, audio_format, jobs=jobs,
        )
        return

    if use_gpu:
        _run_realify_gpu(
            tasks,
            model=model,
            presets_filepath=presets_filepath,
            batch_size=batch_size,
            audio_format=audio_format,
        )
    else:
        _run_realify_cpu(
            tasks,
            model=model,
            presets_filepath=presets_filepath,
            jobs=jobs,
            batch_size=batch_size,
            audio_format=audio_format,
        )

    write_mixtures_for_dataset(
        source_dir, output_dir, audio_format, jobs=jobs,
    )


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
    jobs: int = 1,
):
    """Build mixture per song from stems in the output tree (same procedure as synthesis)."""
    stems_csv = output_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.exists():
        stems_csv = source_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.exists():
        return

    stems = pd.read_csv(stems_csv)
    tasks = build_mixture_tasks(stems, source_dir, output_dir, audio_format)
    if not tasks:
        return

    n_workers = min(max(jobs, 1), len(tasks))
    desc = "Writing mixtures" if n_workers == 1 else f"Writing mixtures ({n_workers} workers)"
    if n_workers == 1:
        for task in tqdm(tasks, desc=desc, unit="song"):
            write_mixture_task(task)
        return

    pool = multiprocessing.Pool(processes=n_workers)
    try:
        for _ in tqdm(
            pool.imap(write_mixture_task, tasks, chunksize=1),
            total=len(tasks),
            desc=desc,
            unit="song",
        ):
            pass
    finally:
        _shutdown_pool(pool)


def build_mixture_tasks(
    stems: pd.DataFrame,
    source_dir: Path,
    output_dir: Path,
    audio_format: str,
) -> list[dict]:
    tasks = []
    for song_path, group in stems.groupby("path"):
        out_song_dir = resolve_output_song_dir(Path(song_path), source_dir, output_dir)
        tasks.append({
            "out_song_dir": str(out_song_dir),
            "tracks": sorted(int(t) for t in group["track"]),
            "audio_format": audio_format,
        })
    return tasks


def write_mixture_task(task: dict) -> str | None:
    out_path = write_mixture_from_song_dir(
        Path(task["out_song_dir"]),
        task["tracks"],
        task["audio_format"],
    )
    return str(out_path) if out_path is not None else None


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
        help="CPU workers for synthesis, CPU realify (small-music), and realify mixture writes.",
    )
    parser.add_argument(
        "--realify-batch-size",
        default=REALIFY_BATCH_SIZE,
        type=int,
        help="SA3 stems per GPU forward pass (default: REALIFY_BATCH_SIZE in shared/config.py).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the realify output directory and re-realify all stems.",
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
        batch_size=args.realify_batch_size,
        audio_format=synthesis_audio_format(args.mp3),
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
