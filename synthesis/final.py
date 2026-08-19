"""Hybrid production synthesis from a per-category recipe."""

from __future__ import annotations

import argparse

from synthesis.audio import synthesis_audio_format
from synthesis.cli_common import add_synthesis_args
from synthesis.mix import print_mix_hint
from synthesis.paths import (
    ablation_raw_dir,
    ablation_realify_dir,
    full_stems_dir,
    full_stems_realify_dir,
)
from shared.repo_symlinks import link_ablations_in_repo
from synthesis.recipe import (
    DEFAULT_RECIPE_PATH,
    load_recipe,
    require_recipe_conflicts_ok,
    scan_recipe_conflicts,
)
from synthesis.synthesize import (
    require_raw_synthesis,
    run_realify_pass,
    run_synthesis,
)

FINAL_CONDITION = "final"
ONLY_PASSES = ("fluidsynth", "ddsp", "realify")


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        prog="synthesis.final",
        description=(
            "Synthesize one mixed stem tree using a per-category recipe. "
            "Passes: Fluidsynth (basic/slakh per track), then MIDI-DDSP / DDSP-Piano, "
            "then SA3 realify for categories whose recipe sets realify."
        ),
    )
    add_synthesis_args(
        parser,
        include_render_mode=False,
        include_realify=False,
        full_default=True,
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
        default=None,
        help="Run a single method pass (fluidsynth, ddsp, or realify) and exit.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Regenerate stems that no longer match the recipe without prompting.",
    )
    return parser.parse_args(args=args, namespace=namespace)


def hybrid_dirs(args) -> tuple[str, str]:
    if args.full:
        return full_stems_dir(args.output_dir), full_stems_realify_dir(args.output_dir)
    return (
        ablation_raw_dir(args.output_dir, FINAL_CONDITION),
        ablation_realify_dir(args.output_dir, FINAL_CONDITION),
    )


def log_recipe_plan(recipe, *, source_dir: str, dest_dir: str) -> None:
    grouped = recipe.pass_categories()
    print(f"Recipe: {recipe.path or '(in-memory)'}")
    print(f"  Fluidsynth categories: {', '.join(grouped['fluidsynth']) or '(none)'}")
    print(f"  MIDI-DDSP categories: {', '.join(grouped['ddsp']) or '(none)'}")
    print(f"  Realify categories: {', '.join(grouped['realify']) or '(none)'}")
    print(f"  Raw stems: {source_dir}")
    print(f"  Realify dest: {dest_dir}")


def main(argv=None):
    args = parse_args(argv)
    recipe = load_recipe(args.recipe)
    args.recipe = recipe
    args.render_mode = "basic"
    args.realify = False
    args.only_pass = getattr(args, "only_pass", None)
    source_dir, dest_dir = hybrid_dirs(args)
    log_recipe_plan(recipe, source_dir=source_dir, dest_dir=dest_dir)

    only = args.only_pass
    run_raw = only in (None, "fluidsynth", "ddsp")
    run_realify = only in (None, "realify")
    audio_format = synthesis_audio_format(args.flac)

    if run_raw:
        if only == "ddsp":
            args.skip_output_reset = True
        run_synthesis(args, source_dir)

    stems_dir = source_dir
    if run_realify:
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
            stems_dir = dest_dir

    link_ablations_in_repo(args.output_dir)
    print_mix_hint(stems_dir, jobs=args.jobs, flac=bool(args.flac))


if __name__ == "__main__":
    main()
