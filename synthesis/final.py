"""Hybrid production synthesis from a per-category recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.config import OUTPUT_DIR, SPDMX_DATASET_DIR_NAME
from synthesis.audio import synthesis_audio_format
from synthesis.cli_common import add_synthesis_args
from synthesis.paths import ablation_raw_dir, spdmx_dataset_dir
from shared.repo_symlinks import link_ablations_in_repo
from synthesis.recipe import (
    DEFAULT_RECIPE_PATH,
    load_recipe,
    require_recipe_conflicts_ok,
    scan_recipe_conflicts,
)
from synthesis.synthesize import (
    require_raw_synthesis,
    run_layout_pass,
    run_realify_pass,
    run_synthesis,
)

FINAL_CONDITION = "final"
ONLY_PASSES = ("layout", "fluidsynth", "ddsp", "realify", "mix")


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        prog="synthesis.final",
        description=(
            "Synthesize the sPDMX dataset using a per-category recipe. "
            f"Writes FLAC stems under {OUTPUT_DIR}/{SPDMX_DATASET_DIR_NAME}/ "
            "(same data/ hashed layout as PDMX). "
            "Run one pass at a time with --only-pass "
            "(layout → fluidsynth → ddsp → realify → mix)."
        ),
    )
    add_synthesis_args(
        parser,
        include_render_mode=False,
        include_realify=False,
        full_default=True,
        flac_default=True,
    )
    parser.add_argument(
        "--recipe",
        default=str(DEFAULT_RECIPE_PATH),
        type=str,
        help="YAML mapping listening category → ablation id (or method/realify/fallback).",
    )
    parser.add_argument(
        "--only-pass",
        choices=list(ONLY_PASSES),
        required=True,
        help=(
            "Required. One method pass: layout (pass 0, mkdir song dirs), "
            "fluidsynth, ddsp, realify, or mix. Do not chain methods in one job."
        ),
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Regenerate stems that no longer match the recipe without prompting.",
    )
    return parser.parse_args(args=args, namespace=namespace)


def hybrid_dirs(args) -> tuple[str, str]:
    """Return (source, dest). Production writes one tree; realify overwrites in place."""
    if args.full:
        dest = spdmx_dataset_dir(args.output_dir)
        return dest, dest
    dest = ablation_raw_dir(args.output_dir, FINAL_CONDITION)
    return dest, dest


def pass_sequence(recipe) -> tuple[str, ...]:
    """Ordered production passes for this recipe (always starts with layout)."""
    steps = ["layout", "fluidsynth"]
    if recipe.uses_ddsp():
        steps.append("ddsp")
    if recipe.uses_realify():
        steps.append("realify")
    steps.append("mix")
    return tuple(steps)


def log_recipe_plan(recipe, *, stems_dir: str, only: str) -> None:
    grouped = recipe.pass_categories()
    plan = pass_sequence(recipe)
    print(f"Recipe: {recipe.path or '(in-memory)'}")
    print(f"  Fluidsynth categories: {', '.join(grouped['fluidsynth']) or '(none)'}")
    print(f"  MIDI-DDSP categories: {', '.join(grouped['ddsp']) or '(none)'}")
    print(f"  Realify categories: {', '.join(grouped['realify']) or '(none)'}")
    print(f"  Stems: {stems_dir}")
    print(f"  Passes: {' → '.join(plan)}")
    print(f"  This job: {only}")


def log_next_pass(recipe, only: str) -> None:
    plan = pass_sequence(recipe)
    if only not in plan:
        return
    idx = plan.index(only)
    if idx + 1 >= len(plan):
        print("All passes complete.", flush=True)
        return
    nxt = plan[idx + 1]
    extra = " -j 8" if nxt in ("fluidsynth", "mix") else ""
    print(f"Next: uv run python -m synthesis.final --only-pass {nxt}{extra}", flush=True)


def run_summable_mix(args, stems_dir: str) -> None:
    from synthesis.mix import normalize_stems_for_dataset

    audio_format = synthesis_audio_format(args.flac)
    print(
        "Normalizing stems in place "
        f"(LUFS + velocity + peak; {audio_format}; mix = sum of stems, no mixture file).",
        flush=True,
    )
    normalize_stems_for_dataset(
        Path(stems_dir),
        Path(stems_dir),
        audio_format=audio_format,
        jobs=args.jobs,
        write_mixture=False,
        pdmx_root=Path(args.dataset_filepath).parent,
        spdmx_output_dir=args.output_dir,
    )


def main(argv=None):
    args = parse_args(argv)
    recipe = load_recipe(args.recipe)
    args.recipe = recipe
    args.render_mode = "basic"
    args.realify = False
    only = args.only_pass
    args.only_pass = only
    source_dir, dest_dir = hybrid_dirs(args)
    log_recipe_plan(recipe, stems_dir=source_dir, only=only)
    audio_format = synthesis_audio_format(args.flac)

    if only != "layout":
        args.skip_output_reset = True

    if only == "layout":
        run_layout_pass(args, source_dir)
    elif only in ("fluidsynth", "ddsp"):
        run_synthesis(args, source_dir)
    elif only == "realify":
        if not recipe.uses_realify():
            print("Realify pass skipped (no category recipe sets realify).")
        else:
            require_raw_synthesis(
                source_dir,
                run_command="uv run python -m synthesis.final --only-pass fluidsynth && "
                "uv run python -m synthesis.final --only-pass ddsp",
                audio_format=audio_format,
            )
            if not args.reset:
                require_recipe_conflicts_ok(
                    scan_recipe_conflicts(
                        dest_dir, recipe, audio_format=audio_format, stage="realify",
                    ),
                    yes=bool(args.yes),
                )
            run_realify_pass(args, source_dir, dest_dir)
    elif only == "mix":
        require_raw_synthesis(
            source_dir,
            run_command="uv run python -m synthesis.final --only-pass fluidsynth",
            audio_format=audio_format,
        )
        run_summable_mix(args, source_dir)

    log_next_pass(recipe, only)
    link_ablations_in_repo(args.output_dir)


if __name__ == "__main__":
    main()
