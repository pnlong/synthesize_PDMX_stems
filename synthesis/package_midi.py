"""Copy dense corrected MIDIs into the sPDMX ``mid/`` tree."""

from __future__ import annotations

import argparse
import shutil
from os.path import dirname
from pathlib import Path

from tqdm import tqdm

from analysis.corrected_midi import resolve_corrected_midi_path, track_map_path_for_midi
from shared.config import OUTPUT_DIR
from synthesis.cli_common import add_synthesis_args
from synthesis.dense_midi import default_corrected_midi_dir
from synthesis.paths import spdmx_dataset_dir
from synthesis.synthesize import prepare_render_dataset


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        prog="synthesis.package_midi",
        description=(
            f"Copy sanitized (dense, register-corrected) MIDIs from "
            f"dev/mid_corrected/ into {OUTPUT_DIR}/SPDMX/mid/, matching PDMX's "
            "mid/<shard>/<shard>/Qm….mid layout. Also copies .mid.track_map.csv sidecars."
        ),
    )
    add_synthesis_args(
        parser,
        include_render_mode=False,
        include_realify=False,
        full_default=True,
        flac_default=True,
    )
    return parser.parse_args(args=args, namespace=namespace)


def package_corrected_midis(
    args,
    dest_root: str | Path | None = None,
) -> tuple[int, int]:
    """Copy corrected MIDIs + track maps. Returns (copied, skipped)."""
    dest_root = Path(dest_root or spdmx_dataset_dir(args.output_dir))
    dest_root.mkdir(parents=True, exist_ok=True)
    pdmx_root = Path(dirname(args.dataset_filepath)).resolve()
    corrected_root = Path(
        getattr(args, "corrected_midi_dir", None)
        or default_corrected_midi_dir(args.output_dir)
    )
    dataset = prepare_render_dataset(args, str(dest_root), register_df=None)
    copied = 0
    skipped = 0
    iterator = dataset.itertuples(index=False)
    if len(dataset) > 1:
        iterator = tqdm(
            dataset.itertuples(index=False),
            total=len(dataset),
            desc="Packaging MIDI",
            unit="song",
        )
    for row in iterator:
        src = resolve_corrected_midi_path(
            row.mid_pdmx,
            pdmx_root=pdmx_root,
            corrected_midi_dir=corrected_root,
        )
        if not src.is_file():
            raise FileNotFoundError(
                f"Corrected MIDI missing: {src}\n"
                "Generate corrected midis first:\n"
                "  uv run python -m analysis.prepare_synthesis --subset all_valid -j 8"
            )
        rel = Path(row.mid).resolve().relative_to(pdmx_root)
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and not args.reset:
            skipped += 1
        else:
            shutil.copy2(src, dest)
            copied += 1
        map_src = track_map_path_for_midi(src)
        map_dest = track_map_path_for_midi(dest)
        if map_src.is_file() and (args.reset or not map_dest.is_file()):
            shutil.copy2(map_src, map_dest)
    print(
        f"Packaged MIDI into {dest_root / 'mid'}: copied {copied}, skipped {skipped}",
        flush=True,
    )
    return copied, skipped


def main(argv=None):
    args = parse_args(argv)
    args.recipe = None
    package_corrected_midis(args)


if __name__ == "__main__":
    main()
