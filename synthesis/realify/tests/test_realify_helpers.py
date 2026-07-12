"""Unit tests for realify task building and device selection."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from synthesis.realify.captions.generate import generate_captions
from synthesis.realify.preset_config import select_preset
from synthesis.realify.realify import (
    build_realify_tasks,
    build_generate_kwargs,
    build_mixture_tasks,
    configure_sa3_runtime,
    copy_metadata_tables,
    iter_realify_batches,
    log_realify_plan,
    realify_stem,
    realify_stems_batch,
    realify_uses_gpu,
    load_model,
    reset_realify_output,
    select_realify_gpu_indices,
    should_use_flash_attention,
    stem_seed,
    write_mixture_task,
    write_mixtures_for_dataset,
    _normalize_generated_audio,
    _run_realify_gpu,
)


def test_realify_uses_gpu_medium_requires_cuda():
    with patch("synthesis.realify.realify.visible_cuda_count", return_value=0):
        with pytest.raises(RuntimeError, match="medium requires a GPU"):
            realify_uses_gpu("medium")


def test_realify_uses_gpu_medium_with_cuda():
    with patch("synthesis.realify.realify.visible_cuda_count", return_value=2):
        assert realify_uses_gpu("medium") is True


def test_realify_uses_gpu_small_music_prefers_cuda():
    with patch("synthesis.realify.realify.visible_cuda_count", return_value=1):
        assert realify_uses_gpu("small-music") is True


def test_realify_uses_gpu_small_music_falls_back_to_cpu():
    with patch("synthesis.realify.realify.visible_cuda_count", return_value=0):
        assert realify_uses_gpu("small-music") is False


def test_build_realify_tasks_skips_existing(tmp_path: Path):
    source_dir = tmp_path / "basic"
    output_dir = tmp_path / "basic_realify"
    song_dir = source_dir / "data" / "song"
    out_song_dir = output_dir / "data" / "song"
    song_dir.mkdir(parents=True)
    out_song_dir.mkdir(parents=True)

    sr = 44100
    sf.write(str(song_dir / "stem_0.flac"), np.zeros(sr), sr, format="FLAC")
    sf.write(str(song_dir / "stem_1.flac"), np.zeros(sr), sr, format="FLAC")
    sf.write(str(out_song_dir / "stem_0.flac"), np.zeros(sr), sr, format="FLAC")

    captions = pd.DataFrame({
        "path": [str(song_dir), str(song_dir)],
        "track": [0, 1],
        "prompt": ["piano", "drums"],
        "is_drum": [False, True],
        "name": [None, None],
    })

    tasks = build_realify_tasks(captions, source_dir, output_dir)
    assert len(tasks) == 1
    assert tasks[0]["out_path"] == str(out_song_dir / "stem_1.flac")


def test_build_realify_tasks_skips_invalid_stem(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "basic"
    output_dir = tmp_path / "basic_realify"
    song_dir = source_dir / "data" / "song"
    song_dir.mkdir(parents=True)

    sr = 44100
    sf.write(str(song_dir / "stem_0.flac"), np.zeros(sr), sr, format="FLAC")
    (song_dir / "stem_1.flac").write_bytes(b"bad")

    captions = pd.DataFrame({
        "path": [str(song_dir), str(song_dir)],
        "track": [0, 1],
        "prompt": ["piano", "drums"],
    })

    monkeypatch.setattr(
        "synthesis.realify.realify.stem_is_valid",
        lambda path: path.name == "stem_0.flac",
    )
    tasks = build_realify_tasks(captions, source_dir, output_dir)
    assert len(tasks) == 1
    assert tasks[0]["stem_path"].endswith("stem_0.flac")


def test_build_realify_tasks_uses_mp3_when_requested(tmp_path: Path):
    from shared.config import PROTOTYPE_AUDIO_FORMAT

    source_dir = tmp_path / "basic"
    output_dir = tmp_path / "basic_realify"
    song_dir = source_dir / "data" / "song"
    out_song_dir = output_dir / "data" / "song"
    song_dir.mkdir(parents=True)
    out_song_dir.mkdir(parents=True)

    sr = 44100
    sf.write(str(song_dir / "stem_0.mp3"), np.zeros(sr), sr, format="MP3")

    captions = pd.DataFrame({
        "path": [str(song_dir)],
        "track": [0],
        "prompt": ["piano"],
    })

    tasks = build_realify_tasks(
        captions, source_dir, output_dir, audio_format=PROTOTYPE_AUDIO_FORMAT,
    )
    assert len(tasks) == 1
    assert tasks[0]["stem_path"].endswith("stem_0.mp3")
    assert tasks[0]["out_path"].endswith("stem_0.mp3")


def test_realify_stem_passes_init_audio_as_sample_rate_then_tensor(tmp_path: Path, monkeypatch):
    import torch

    from shared.config import REALIFY_INIT_NOISE_LEVEL, REALIFY_STEPS, SAMPLE_RATE

    captured = {}

    class FakeModel:
        model_config = {"sample_size": 5292032}
        model = type("M", (), {"sample_rate": 44100})()

        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros(1, 1, 100)

    monkeypatch.setattr(
        "synthesis.realify.realify.load_stem",
        lambda path: torch.ones(1, 50),
    )
    monkeypatch.setattr(
        "synthesis.realify.realify.write_audio",
        lambda audio, path, audio_format: path,
    )

    out = tmp_path / "stem_0.flac"
    realify_stem(
        init_audio_path=tmp_path / "in.flac",
        output_path=out,
        prompt="piano",
        preset={"steps": 12, "cfg_scale": 1.0, "init_noise_level": 0.55},
        model=FakeModel(),
        duration_seconds=1.0,
        seed=42,
    )

    sr, audio = captured["init_audio"]
    assert sr == SAMPLE_RATE
    assert torch.is_tensor(audio)
    assert captured["init_noise_level"] == 0.55
    assert captured["steps"] == 12
    assert captured["cfg_scale"] == 1.0
    assert captured["seed"] == 42
    assert captured["sample_size"] == 5292032
    assert captured["batch_size"] == 1
    assert captured["disable_tqdm"] is True


def test_build_generate_kwargs_uses_default_init_noise_level():
    from shared.config import REALIFY_INIT_NOISE_LEVEL
    from synthesis.realify.realify import build_generate_kwargs

    class FakeModel:
        model_config = {"sample_size": 1}

    kwargs = build_generate_kwargs(
        preset={"steps": 8, "cfg_scale": 1.0},
        model=FakeModel(),
        prompt="piano",
        duration_seconds=1.0,
        init_audio=(44100, None),
        seed=1,
    )
    assert kwargs["init_noise_level"] == REALIFY_INIT_NOISE_LEVEL


def test_build_generate_kwargs_passes_negative_prompt():
    from synthesis.realify.realify import build_generate_kwargs

    class FakeModel:
        model_config = {"sample_size": 1}

    kwargs = build_generate_kwargs(
        preset={
            "steps": 8,
            "cfg_scale": 1.0,
            "negative_prompt": "synthetic, midi",
        },
        model=FakeModel(),
        prompt="piano",
        duration_seconds=1.0,
        init_audio=(44100, None),
        seed=1,
    )
    assert kwargs["negative_prompt"] == "synthetic, midi"


def test_load_model_passes_explicit_cuda_device(monkeypatch):
    captured = {}

    class FakeStableAudioModel:
        @staticmethod
        def from_pretrained(model_name, device=None, model_half=True):
            captured["model_name"] = model_name
            captured["device"] = device
            return f"model-on-{device}"

    monkeypatch.setattr(
        "stable_audio_3.StableAudioModel",
        FakeStableAudioModel,
    )
    monkeypatch.setattr("synthesis.realify.realify.configure_sa3_runtime", lambda **kwargs: None)

    model = load_model("medium", device_index=2)
    assert model == "model-on-cuda:2"
    assert captured["device"] == "cuda:2"


def test_select_preset_merges_default_and_instrument(tmp_path: Path):
    presets = {
        "default": {"steps": 8, "cfg_scale": 1.0, "init_noise_level": 0.45, "prompt_variant": "current"},
        "categories": {
            "drums": {"steps": 10},
            "piano": {"cfg_scale": 0.9},
        },
        "routing": [
            {"category": "drums", "is_drum": True},
            {"category": "piano", "name_keywords": ["piano"]},
        ],
    }
    drum_row = pd.Series({"is_drum": True, "name": "Kick"})
    piano_row = pd.Series({"is_drum": False, "name": "Piano"})
    assert select_preset(presets, drum_row) == {
        "steps": 10,
        "cfg_scale": 1.0,
        "init_noise_level": 0.45,
        "prompt_variant": "current",
    }
    assert select_preset(presets, piano_row) == {
        "steps": 8,
        "cfg_scale": 0.9,
        "init_noise_level": 0.45,
        "prompt_variant": "current",
    }


def test_build_realify_tasks_includes_seed(tmp_path: Path):
    source_dir = tmp_path / "basic"
    output_dir = tmp_path / "basic_realify"
    song_dir = source_dir / "data" / "song"
    song_dir.mkdir(parents=True)

    sr = 44100
    sf.write(str(song_dir / "stem_0.flac"), np.zeros(sr), sr, format="FLAC")

    captions = pd.DataFrame({
        "path": [str(song_dir)],
        "track": [0],
        "prompt": ["piano"],
    })

    tasks = build_realify_tasks(captions, source_dir, output_dir, sample_seed=7)
    assert len(tasks) == 1
    assert tasks[0]["seed"] == stem_seed(7, str(song_dir), 0)


def test_should_use_flash_attention_requires_ampere(monkeypatch):
    monkeypatch.setattr(
        "synthesis.realify.realify.visible_cuda_count",
        lambda: 1,
    )

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_capability(device_index):
            assert device_index == 0
            return (7, 5)

    import torch

    monkeypatch.setattr(torch, "cuda", FakeCuda)
    assert should_use_flash_attention(0) is False


def test_realify_stem_chunks_long_stems(tmp_path: Path, monkeypatch):
    import torch

    captured = []

    class FakeModel:
        model_config = {"sample_size": 44100 * 10}
        model = type("M", (), {"sample_rate": 44100})()

        def generate(self, **kwargs):
            captured.append(kwargs)
            duration = kwargs["duration"]
            n_samples = int(duration * 44100)
            return torch.full((1, 1, n_samples), 0.5)

    long_waveform = torch.ones(1, 44100 * 25)

    monkeypatch.setattr(
        "synthesis.realify.realify.load_stem",
        lambda path: long_waveform,
    )
    monkeypatch.setattr(
        "synthesis.realify.realify.write_audio",
        lambda audio, path, audio_format: path,
    )

    out = tmp_path / "stem_0.flac"
    realify_stem(
        init_audio_path=tmp_path / "in.flac",
        output_path=out,
        prompt="piano",
        preset={"steps": 8, "cfg_scale": 1.0},
        model=FakeModel(),
        duration_seconds=25.0,
        seed=42,
    )

    assert len(captured) > 1
    assert all(call["duration"] <= 10 - 6.0 + 1e-6 for call in captured)


def test_run_realify_gpu_uses_spawn_pool(tmp_path: Path, monkeypatch):
    captured = {}

    class FakeContext:
        def Pool(self, **kwargs):
            captured["pool_kwargs"] = kwargs
            return self

        def close(self):
            captured["pool_closed"] = True

        def join(self):
            captured["pool_joined"] = True

        def imap(self, func, tasks, chunksize=1):
            captured["imap_chunksize"] = chunksize
            captured["imap_func"] = func.__name__
            return []

    fake_ctx = FakeContext()

    monkeypatch.setattr(
        "synthesis.realify.realify.multiprocessing.get_context",
        lambda method: fake_ctx if method == "spawn" else pytest.fail(f"unexpected context {method}"),
    )
    monkeypatch.setattr(
        "synthesis.realify.realify.select_realify_gpu_indices",
        lambda **kwargs: [0, 1],
    )
    monkeypatch.setattr(
        "synthesis.realify.realify.gpu_memory_snapshot",
        lambda device_index: (20.0, 24.0, f"GPU-{device_index}"),
    )

    _run_realify_gpu(
        [{"row": {"prompt": "p", "path": "/p", "track": 0}, "stem_path": "/s", "out_path": "/o", "duration": 1.0, "seed": 1, "audio_format": "flac"}] * 2,
        model="medium",
        presets_filepath=tmp_path / "presets" / "categories.yaml",
        batch_size=1,
        audio_format="flac",
    )

    assert captured["pool_kwargs"]["processes"] == 2
    assert captured["pool_kwargs"]["initargs"] == ("medium", str(tmp_path / "presets" / "categories.yaml"), 1)
    assert captured["imap_func"] == "_realify_gpu_worker_shard"
    assert captured["imap_chunksize"] == 1
    assert captured["pool_closed"] is True
    assert captured["pool_joined"] is True


def test_select_realify_gpu_indices_filters_low_memory(monkeypatch):
    monkeypatch.setattr("synthesis.realify.realify.visible_cuda_count", lambda: 3)
    monkeypatch.setattr(
        "synthesis.realify.realify.gpu_memory_snapshot",
        lambda device_index: {
            0: (20.0, 24.0, "RTX 3090"),
            1: (2.0, 24.0, "RTX 3090 busy"),
            2: (11.5, 11.5, "RTX 2080 Ti"),
        }[device_index],
    )

    assert select_realify_gpu_indices(min_free_gb=10.0, log_skips=False) == [0, 2]


def test_build_generate_kwargs_enables_chunked_decode():
    class FakeModel:
        model_config = {"sample_size": 123}

    kwargs = build_generate_kwargs(
        preset={},
        model=FakeModel(),
        prompt="test",
        duration_seconds=10.0,
        init_audio=(44100, None),
        seed=1,
    )
    assert kwargs["chunked_decode"] is True
    assert kwargs["batch_size"] == 1


def test_iter_realify_batches_groups_by_preset_and_size(monkeypatch):
    class FakeModel:
        model_config = {"sample_size": 44100 * 120}
        model = type("M", (), {"sample_rate": 44100})()

    presets = {
        "default": {"steps": 8, "cfg_scale": 1.0, "init_noise_level": 0.45, "prompt_variant": "current"},
        "categories": {
            "drums": {"steps": 10, "cfg_scale": 1.0},
        },
        "routing": [
            {"category": "drums", "is_drum": True},
            {"category": "piano", "name_keywords": ["piano"]},
        ],
    }
    tasks = [
        {"row": {"prompt": "a", "is_drum": False, "name": "Piano"}, "duration": 10.0},
        {"row": {"prompt": "b", "is_drum": False, "name": "Piano"}, "duration": 12.0},
        {"row": {"prompt": "c", "is_drum": True, "name": "Kick"}, "duration": 8.0},
        {"row": {"prompt": "d", "is_drum": False, "name": "Piano"}, "duration": 400.0},
    ]
    monkeypatch.setattr(
        "synthesis.realify.realify.task_needs_chunking",
        lambda task, model: task["duration"] > 120,
    )

    batches = list(iter_realify_batches(tasks, FakeModel(), presets, batch_size=2))
    assert [len(batch) for batch in batches] == [2, 1, 1]


def test_realify_stems_batch_calls_generate_once(tmp_path: Path, monkeypatch):
    import torch

    calls = []

    class FakeModel:
        model_config = {"sample_size": 123}
        model = type("M", (), {"sample_rate": 44100})()

        def generate(self, **kwargs):
            calls.append(kwargs)
            batch_size = kwargs["batch_size"]
            return torch.zeros(batch_size, 2, 50)

    presets = {"default": {"steps": 8, "cfg_scale": 1.0}}
    tasks = [
        {
            "row": {"prompt": "a", "is_drum": False, "name": "Piano"},
            "stem_path": str(tmp_path / "a.flac"),
            "out_path": str(tmp_path / "out_a.flac"),
            "duration": 1.0,
            "seed": 1,
            "audio_format": "flac",
        },
        {
            "row": {"prompt": "b", "is_drum": False, "name": "Piano"},
            "stem_path": str(tmp_path / "b.flac"),
            "out_path": str(tmp_path / "out_b.flac"),
            "duration": 2.0,
            "seed": 2,
            "audio_format": "flac",
        },
    ]
    monkeypatch.setattr(
        "synthesis.realify.realify.load_stem",
        lambda path: torch.ones(2, 40) if "b" in str(path) else torch.ones(2, 30),
    )
    monkeypatch.setattr(
        "synthesis.realify.realify.write_audio",
        lambda audio, path, audio_format: path,
    )

    realify_stems_batch(tasks, model=FakeModel(), presets=presets, audio_format="flac")
    assert len(calls) == 1
    assert calls[0]["batch_size"] == 2
    assert calls[0]["prompt"] == ["a", "b"]
    assert calls[0]["seed"] == [1, 2]
    assert len(calls[0]["init_audio"]) == 2


def test_write_mixtures_for_dataset_uses_pool(tmp_path: Path, monkeypatch):
    captured = {}

    class FakePool:
        def __init__(self, processes):
            captured["processes"] = processes

        def imap(self, func, tasks, chunksize=1):
            captured["n_tasks"] = len(tasks)
            captured["chunksize"] = chunksize
            return [func(task) for task in tasks]

        def close(self):
            captured["closed"] = True

        def join(self):
            captured["joined"] = True

    monkeypatch.setattr("synthesis.realify.realify.multiprocessing.Pool", FakePool)
    monkeypatch.setattr(
        "synthesis.realify.realify.write_mixture_task",
        lambda task: task["out_song_dir"],
    )

    source_dir = tmp_path / "basic"
    output_dir = tmp_path / "basic_realify"
    song_a = source_dir / "data" / "song_a"
    song_b = source_dir / "data" / "song_b"
    song_a.mkdir(parents=True)
    song_b.mkdir(parents=True)
    (output_dir / "data" / "song_a").mkdir(parents=True)
    (output_dir / "data" / "song_b").mkdir(parents=True)

    stems = pd.DataFrame({
        "path": [str(song_a), str(song_b)],
        "track": [0, 0],
    })
    stems.to_csv(output_dir / "stems.csv", index=False)

    write_mixtures_for_dataset(source_dir, output_dir, jobs=4)
    assert captured["processes"] == 2
    assert captured["n_tasks"] == 2
    assert captured["chunksize"] == 1


def test_build_mixture_tasks_resolves_output_dirs(tmp_path: Path):
    source_dir = tmp_path / "basic"
    output_dir = tmp_path / "basic_realify"
    song_dir = source_dir / "data" / "song"
    song_dir.mkdir(parents=True)

    stems = pd.DataFrame({
        "path": [str(song_dir), str(song_dir)],
        "track": [1, 0],
    })
    tasks = build_mixture_tasks(stems, source_dir, output_dir, "flac")
    assert len(tasks) == 1
    assert tasks[0]["tracks"] == [0, 1]
    assert tasks[0]["out_song_dir"].endswith("basic_realify/data/song")


def test_normalize_generated_audio_respects_stem_channels(monkeypatch):
    import torch

    stereo = torch.tensor([[[0.0, 2.0], [1.0, 3.0]]])

    monkeypatch.setattr("synthesis.audio.STEM_CHANNELS", 1)
    mono = _normalize_generated_audio(stereo)
    assert mono.shape == (1, 2)
    assert mono[0, 0] == 0.5
    assert mono[0, 1] == 2.5

    monkeypatch.setattr("synthesis.audio.STEM_CHANNELS", 2)
    kept = _normalize_generated_audio(stereo)
    assert kept.shape == (2, 2)
    assert kept[0, 0] == 0.0
    assert kept[1, 1] == 3.0


def test_run_realify_reset_clears_existing_outputs(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "basic"
    output_dir = tmp_path / "basic_realify"
    song_dir = source_dir / "data" / "song"
    out_song_dir = output_dir / "data" / "song"
    song_dir.mkdir(parents=True)
    out_song_dir.mkdir(parents=True)

    import numpy as np
    import soundfile as sf

    sr = 44100
    sf.write(str(song_dir / "stem_0.flac"), np.zeros(sr), sr, format="FLAC")
    sf.write(str(out_song_dir / "stem_0.flac"), np.zeros(sr), sr, format="FLAC")

    pd.DataFrame({"path": [str(song_dir)], "n_tracks": [1]}).to_csv(
        source_dir / "data.csv", index=False
    )
    pd.DataFrame({
        "path": [str(song_dir)],
        "track": [0],
        "program": [0],
        "is_drum": [False],
        "name": ["Piano"],
        "has_lyrics": [False],
    }).to_csv(source_dir / "stems.csv", index=False)

    tasks_before_reset = build_realify_tasks(
        generate_captions(source_dir),
        source_dir,
        output_dir,
    )
    assert tasks_before_reset == []

    reset_realify_output(output_dir)
    copy_metadata_tables(source_dir, output_dir)
    tasks_after_reset = build_realify_tasks(
        generate_captions(source_dir),
        source_dir,
        output_dir,
    )
    assert len(tasks_after_reset) == 1


def test_log_realify_plan_reports_no_tasks(capsys):
    log_realify_plan(
        source_dir=Path("/src"),
        output_dir=Path("/out"),
        model="medium",
        n_tasks=0,
        n_captions=10,
        use_gpu=True,
    )
    out = capsys.readouterr().out
    assert "stems queued: 0 of 10" in out
    assert "skipping SA3" in out


def test_configure_sa3_runtime_disables_flash_attention_on_pre_ampere(monkeypatch):
    monkeypatch.setattr(
        "synthesis.realify.realify.should_use_flash_attention",
        lambda device_index=None: False,
    )
    patched = {}

    def fake_patch():
        patched["called"] = True

    monkeypatch.setattr(
        "synthesis.realify.realify._patch_sa3_disable_flash_attention",
        fake_patch,
    )
    monkeypatch.setattr(
        "torch.cuda.get_device_capability",
        lambda device_index: (7, 5),
    )
    configure_sa3_runtime(device_index=0)
    assert patched["called"] is True
