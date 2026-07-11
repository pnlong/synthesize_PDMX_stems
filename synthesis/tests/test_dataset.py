"""Tests for ablation paths and dataset sampling."""

import pandas as pd

from shared.config import (
    ABLATION_SAMPLE_SIZE,
    DEV_DIR_NAME,
    OUTPUT_DIR,
    SPDMX_DATASET_DIR_NAME,
    STEMS_DIR_NAME,
    STEMS_REALIFY_DIR_NAME,
)
from synthesis.dataset import prepare_ablation_dataset, prepare_full_dataset
from synthesis.paths import (
    ablation_dir,
    ablation_raw_dir,
    ablation_realify_dir,
    ablations_root,
    condition_name,
    dev_root,
    full_stems_dir,
    full_stems_realify_dir,
    song_lengths_dir,
    spdmx_dataset_dir,
)


def _fake_pdmx(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame({
        "path": [f"/p/{i}.json" for i in range(n)],
        "subset:rated_deduplicated": [i % 2 == 0 for i in range(n)],
    })


def test_dev_root():
    assert dev_root("/out") == f"/out/{DEV_DIR_NAME}"


def test_ablation_paths():
    assert ablations_root(OUTPUT_DIR).endswith(f"/{DEV_DIR_NAME}/ablations")
    assert ablation_dir(OUTPUT_DIR, "basic") == f"{OUTPUT_DIR}/{DEV_DIR_NAME}/ablations/basic"
    assert condition_name("basic", realify=True) == "basic_realify"


def test_ablation_output_dirs():
    assert ablation_raw_dir("/out", "basic") == f"/out/{DEV_DIR_NAME}/ablations/basic"
    assert ablation_realify_dir("/out", "slakh") == f"/out/{DEV_DIR_NAME}/ablations/slakh_realify"


def test_full_stem_dirs():
    assert full_stems_dir("/out") == f"/out/{DEV_DIR_NAME}/{STEMS_DIR_NAME}"
    assert full_stems_realify_dir("/out") == f"/out/{DEV_DIR_NAME}/{STEMS_REALIFY_DIR_NAME}"


def test_song_lengths_dir():
    assert song_lengths_dir("/out") == f"/out/{DEV_DIR_NAME}/analysis/song_lengths"


def test_spdmx_dataset_dir():
    assert spdmx_dataset_dir("/out") == f"/out/{SPDMX_DATASET_DIR_NAME}"


def test_ablation_sample_size_and_seed():
    df = prepare_ablation_dataset(_fake_pdmx(), sample_size=10, sample_seed=0)
    assert len(df) == 10
    df2 = prepare_ablation_dataset(_fake_pdmx(), sample_size=10, sample_seed=0)
    assert list(df["path"]) == list(df2["path"])


def test_ablation_filters_subset():
    df = prepare_ablation_dataset(_fake_pdmx(), sample_size=ABLATION_SAMPLE_SIZE)
    assert df["subset:rated_deduplicated"].all()


def test_full_dataset_keeps_all_rows():
    df = prepare_full_dataset(_fake_pdmx(50))
    assert len(df) == 50
