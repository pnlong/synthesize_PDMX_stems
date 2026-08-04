"""Peak-normalize stems so they remain linearly summable.

Synthesis / realify write stems (typically LUFS-normalized). Run this afterward to:

1. Loudness-normalize stems (−23 LUFS)
2. Sum in memory and compute the Slakh-style anti-clip peak gain
3. Write stems with that same peak gain (overwrites by default; use ``--no-overwrite``)
4. Optionally write ``mixture.*`` with ``--write-mixture`` (otherwise mix = sum(stems))

Example::

    uv run python -m synthesis.mix --render-mode basic -j 8
    uv run python -m synthesis.mix --stems-dir /path/to/ablation --no-overwrite --write-mixture -j 8
"""

from __future__ import annotations

import argparse
import multiprocessing
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from shared.config import (
    DATA_DIR_NAME,
    DEFAULT_AUDIO_FORMAT,
    OUTPUT_DIR,
    RENDER_MODE_BASIC,
    RENDER_MODES,
    STEMS_FILE_NAME,
)
from synthesis.audio import (
    normalize_stems_in_song_dir,
    synthesis_audio_format,
)
from synthesis.cli_common import add_audio_format_arg
from synthesis.paths import (
    ablation_raw_dir,
    ablation_realify_dir,
    full_stems_dir,
    full_stems_realify_dir,
    resolve_output_song_dir,
)


def normalize_song_task(task: dict) -> str | None:
    peak_gain = normalize_stems_in_song_dir(
        Path(task["song_dir"]),
        task["tracks"],
        task["audio_format"],
        dest_song_dir=Path(task["out_song_dir"]),
        write_mixture=bool(task.get("write_mixture", False)),
    )
    return task["out_song_dir"] if peak_gain is not None else None


# Back-compat alias
write_mixture_task = normalize_song_task


def build_mixture_tasks(
    stems: pd.DataFrame,
    source_dir: Path,
    output_dir: Path,
    audio_format: str,
    *,
    write_mixture: bool = False,
) -> list[dict]:
    tasks = []
    for song_path, group in stems.groupby("path"):
        src_song_dir = Path(song_path)
        out_song_dir = resolve_output_song_dir(src_song_dir, source_dir, output_dir)
        tasks.append({
            "song_dir": str(src_song_dir),
            "out_song_dir": str(out_song_dir),
            "tracks": sorted(int(t) for t in group["track"]),
            "audio_format": audio_format,
            "write_mixture": write_mixture,
        })
    return tasks


def _shutdown_pool(pool) -> None:
    pool.close()
    pool.join()


def _remap_path_prefix(value: str, source_dir: Path, output_dir: Path) -> str:
    source_prefix = str(source_dir)
    if value.startswith(source_prefix):
        return str(output_dir) + value[len(source_prefix):]
    return value


def copy_metadata_tables(source_dir: Path, output_dir: Path) -> None:
    """Copy data/stems CSVs into ``output_dir``, remapping absolute song paths."""
    if source_dir.resolve() == output_dir.resolve():
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    stems_src = source_dir / f"{STEMS_FILE_NAME}.csv"
    if stems_src.exists():
        stems = pd.read_csv(stems_src)
        if "path" in stems.columns:
            stems["path"] = stems["path"].map(
                lambda p: _remap_path_prefix(str(p), source_dir, output_dir)
            )
        stems.to_csv(output_dir / f"{STEMS_FILE_NAME}.csv", index=False)

    data_src = source_dir / f"{DATA_DIR_NAME}.csv"
    if data_src.exists():
        data = pd.read_csv(data_src)
        if "path" in data.columns:
            data["path"] = data["path"].map(
                lambda p: _remap_path_prefix(str(p), source_dir, output_dir)
            )
        data.to_csv(output_dir / f"{DATA_DIR_NAME}.csv", index=False)

    for name in ("ddsp_routing.csv",):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def default_dest_dir(stems_dir: Path) -> Path:
    """Sibling directory for non-overwrite writes: ``{name}_summable``."""
    return stems_dir.parent / f"{stems_dir.name}_summable"


def confirm_overwrite(stems_dir: Path) -> bool:
    prompt = (
        f"This will OVERWRITE existing stems in:\n  {stems_dir}\n"
        "Are you sure you want to overwrite? [y/N] "
    )
    try:
        reply = input(prompt).strip().lower()
    except EOFError:
        return False
    return reply in ("y", "yes")


def normalize_stems_for_dataset(
    source_dir: Path,
    output_dir: Path,
    audio_format: str = DEFAULT_AUDIO_FORMAT,
    jobs: int = 1,
    *,
    write_mixture: bool = False,
):
    """Peak-normalize stems so they remain linearly summable.

    Loads stems from ``source_dir`` (via stems.csv paths) and writes to the
    mirrored tree under ``output_dir``. When ``source_dir == output_dir``, stems
    are overwritten in place.
    """
    stems_csv = source_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.exists():
        stems_csv = output_dir / f"{STEMS_FILE_NAME}.csv"
    if not stems_csv.exists():
        return

    if source_dir.resolve() != output_dir.resolve():
        copy_metadata_tables(source_dir, output_dir)

    stems = pd.read_csv(stems_csv)
    # Tasks always load from original stem paths in this CSV.
    tasks = build_mixture_tasks(
        stems, source_dir, output_dir, audio_format, write_mixture=write_mixture,
    )
    if not tasks:
        return

    n_workers = min(max(jobs, 1), len(tasks))
    action = "Normalizing stems" + (" + writing mixtures" if write_mixture else "")
    desc = action if n_workers == 1 else f"{action} ({n_workers} workers)"
    if n_workers == 1:
        for task in tqdm(tasks, desc=desc, unit="song"):
            normalize_song_task(task)
        return

    pool = multiprocessing.Pool(processes=n_workers)
    try:
        for _ in tqdm(
            pool.imap(normalize_song_task, tasks, chunksize=1),
            total=len(tasks),
            desc=desc,
            unit="song",
        ):
            pass
    finally:
        _shutdown_pool(pool)


# Back-compat alias
write_mixtures_for_dataset = normalize_stems_for_dataset


def resolve_stems_dir(
    *,
    stems_dir: str | None = None,
    output_dir: str = OUTPUT_DIR,
    render_mode: str = RENDER_MODE_BASIC,
    realify: bool = False,
    full: bool = False,
) -> Path:
    """Resolve the stem tree to normalize, mirroring synthesize output layout."""
    if stems_dir is not None:
        return Path(stems_dir)
    if full:
        return Path(
            full_stems_realify_dir(output_dir) if realify else full_stems_dir(output_dir)
        )
    return Path(
        ablation_realify_dir(output_dir, render_mode)
        if realify
        else ablation_raw_dir(output_dir, render_mode)
    )


def mix_command(
    stems_dir: str | Path,
    *,
    jobs: int | None = None,
    flac: bool = False,
) -> str:
    """CLI string to print after a stems-only synthesis/realify run."""
    n_jobs = jobs if jobs is not None else max(1, int(multiprocessing.cpu_count() / 4))
    cmd = f"uv run python -m synthesis.mix --stems-dir {stems_dir} -j {n_jobs}"
    if flac:
        cmd += " --flac"
    return cmd


def print_mix_hint(
    stems_dir: str | Path,
    *,
    jobs: int | None = None,
    flac: bool = False,
) -> None:
    print(
        "\nStems written with per-stem LUFS only. "
        "To peak-normalize them so they remain linearly summable "
        "(mix = sum of stems by default), run:",
        flush=True,
    )
    print(f"  {mix_command(stems_dir, jobs=jobs, flac=flac)}", flush=True)
    print(
        "  # Preview without overwriting + write mixture files:\n"
        f"  {mix_command(stems_dir, jobs=jobs, flac=flac)} --no-overwrite --write-mixture",
        flush=True,
    )


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            "Peak-normalize stems (LUFS + uniform anti-clip from the stem sum) "
            "so released stems remain linearly summable."
        ),
    )
    parser.add_argument(
        "--stems-dir",
        default=None,
        type=str,
        help="Stem tree to normalize (overrides --render-mode / --realify / --full).",
    )
    parser.add_argument("-o", "--output_dir", default=OUTPUT_DIR, type=str)
    parser.add_argument(
        "--render-mode",
        default=RENDER_MODE_BASIC,
        choices=list(RENDER_MODES),
        help="Ablation/full stem tree to resolve when --stems-dir is omitted.",
    )
    parser.add_argument("--realify", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Normalize full PDMX stems tree instead of an ablation.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help=(
            "Do not overwrite source stems; write normalized stems to --dest-dir "
            f"(default: sibling {{name}}_summable)."
        ),
    )
    parser.add_argument(
        "--dest-dir",
        default=None,
        type=str,
        help="Destination tree when using --no-overwrite (default: {stems-dir}_summable).",
    )
    parser.add_argument(
        "--write-mixture",
        action="store_true",
        help="Also write mixture.* beside the (normalized) stems.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the overwrite confirmation prompt.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        "--workers",
        default=int(multiprocessing.cpu_count() / 4),
        type=int,
        help="CPU workers for stem normalization.",
    )
    add_audio_format_arg(parser)
    return parser.parse_args(args)


def main(args=None) -> None:
    opts = parse_args(args)
    stems_dir = resolve_stems_dir(
        stems_dir=opts.stems_dir,
        output_dir=opts.output_dir,
        render_mode=opts.render_mode,
        realify=opts.realify,
        full=opts.full,
    )
    if not stems_dir.is_dir():
        raise SystemExit(f"Stem directory not found: {stems_dir}")

    overwrite = not bool(opts.no_overwrite)
    if overwrite:
        dest_dir = stems_dir
        if not opts.yes and not confirm_overwrite(stems_dir):
            raise SystemExit("Aborted (stems not overwritten).")
    else:
        dest_dir = Path(opts.dest_dir) if opts.dest_dir else default_dest_dir(stems_dir)
        if dest_dir.resolve() == stems_dir.resolve():
            raise SystemExit(
                "--no-overwrite requires a different --dest-dir than the source stems tree."
            )

    audio_format = synthesis_audio_format(opts.flac)
    mode = "overwrite in place" if overwrite else f"write to {dest_dir}"
    mix_note = ", writing mixtures" if opts.write_mixture else ""
    print(
        f"Normalizing stems for summability ({mode}{mix_note}; "
        f"{audio_format}, -j {opts.jobs})",
        flush=True,
    )
    normalize_stems_for_dataset(
        stems_dir,
        dest_dir,
        audio_format=audio_format,
        jobs=opts.jobs,
        write_mixture=bool(opts.write_mixture),
    )
    print(f"Done. Output: {dest_dir}", flush=True)


if __name__ == "__main__":
    main()
