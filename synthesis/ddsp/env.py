"""Locate the isolated TF/DDSP Python interpreter."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from synthesis.ddsp.config import DDSP_PYTHON, DDSP_VENV_DIR


class DdspEnvError(RuntimeError):
    """Raised when the neural-DDSP TensorFlow environment is missing or unusable."""


def ddsp_python_executable() -> Path:
    """Return the Python binary for the isolated DDSP venv.

    Override with ``SPDMX_DDSP_PYTHON`` (full path) or ``SPDMX_DDSP_VENV``.
    """
    override = os.environ.get("SPDMX_DDSP_PYTHON")
    if override:
        path = Path(override)
        if not path.is_file():
            raise DdspEnvError(
                f"SPDMX_DDSP_PYTHON={override} is not an executable file. "
                "See SETUP.md Track C (neural DDSP)."
            )
        return path

    if DDSP_PYTHON.is_file():
        return DDSP_PYTHON

    # Fall back to `python` on PATH only if explicitly allowed (dev/tests).
    if os.environ.get("SPDMX_DDSP_ALLOW_SYSTEM_PYTHON") == "1":
        found = shutil.which("python") or shutil.which("python3")
        if found:
            return Path(found)

    raise DdspEnvError(
        f"Neural DDSP venv not found at {DDSP_VENV_DIR}. "
        "Create it with the Track C steps in SETUP.md "
        "(Linux x86_64; not supported on Apple Silicon for midi-ddsp)."
    )


def _venv_site_packages(python_exe: Path) -> Path | None:
    """Best-effort site-packages for the DDSP venv (without importing that Python).

    Do **not** ``Path.resolve()`` the interpreter: uv venvs symlink
    ``.venv-ddsp/bin/python`` to a shared CPython install, and resolving would
    point at the wrong ``lib/`` tree.
    """
    exe = Path(python_exe)
    bin_dir = exe.parent
    venv_root = bin_dir.parent if bin_dir.name == "bin" else bin_dir
    lib = venv_root / "lib"
    if not lib.is_dir():
        return None
    for child in sorted(lib.glob("python*/site-packages"), reverse=True):
        if child.is_dir():
            return child
    return None


def nvidia_cuda_lib_dirs(python_exe: Path | None = None) -> list[str]:
    """Return ``nvidia/*/lib`` dirs from the DDSP venv (pip CUDA 12 wheels).

    TF 2.15 needs CUDA 12.x + cuDNN 8; these come from ``nvidia-*-cu12`` pip
    packages so the host driver CUDA 13 does not have to provide matching libs.
    """
    python_exe = python_exe or ddsp_python_executable()
    site_packages = _venv_site_packages(python_exe)
    if site_packages is None:
        return []
    libs: list[str] = []
    for lib_dir in sorted(site_packages.glob("nvidia/*/lib")):
        libs.append(str(lib_dir))
    return libs


def parse_ddsp_gpu_ids(
    *,
    force_cpu: bool | None = None,
    cuda_visible: str | None = None,
    spdmx_cuda_visible: str | None = None,
) -> list[str]:
    """Return logical GPU id strings for one serve worker each.

    - ``SPDMX_DDSP_FORCE_CPU=1`` → ``["-1"]`` (single CPU worker)
    - Else prefer ``SPDMX_DDSP_CUDA_VISIBLE_DEVICES``, then ``CUDA_VISIBLE_DEVICES``
    - Default ``["0"]``
    """
    if force_cpu is None:
        force_cpu = os.environ.get("SPDMX_DDSP_FORCE_CPU") == "1"
    if force_cpu:
        return ["-1"]

    if spdmx_cuda_visible is None:
        spdmx_cuda_visible = os.environ.get("SPDMX_DDSP_CUDA_VISIBLE_DEVICES")
    if cuda_visible is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    raw = spdmx_cuda_visible if spdmx_cuda_visible is not None else cuda_visible
    if raw is None or str(raw).strip() == "":
        return ["0"]
    raw = str(raw).strip()
    if raw == "-1":
        return ["-1"]
    ids = [part.strip() for part in raw.split(",") if part.strip() != ""]
    return ids or ["0"]


def ddsp_worker_env(
    base: dict[str, str] | None = None,
    *,
    cuda_visible_devices: str | None = None,
) -> dict[str, str]:
    """Environment for the TF worker: CUDA 12 pip libs on LD_LIBRARY_PATH.

    GPU is on by default. Set ``SPDMX_DDSP_FORCE_CPU=1`` to hide devices.
    Optional ``SPDMX_DDSP_CUDA_VISIBLE_DEVICES`` (default ``0``) picks a GPU.
    Pass ``cuda_visible_devices`` to pin a single pool worker to one device.
    """
    env = dict(base if base is not None else os.environ)
    python = ddsp_python_executable()
    libs = nvidia_cuda_lib_dirs(python)
    if libs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            os.pathsep.join([*libs, existing]) if existing else os.pathsep.join(libs)
        )

    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    elif os.environ.get("SPDMX_DDSP_FORCE_CPU") == "1":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        # Prefer an explicit picker; default to first GPU so we don't claim all cards.
        picker = os.environ.get("SPDMX_DDSP_CUDA_VISIBLE_DEVICES")
        if picker is not None:
            env["CUDA_VISIBLE_DEVICES"] = picker
        else:
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def assert_ddsp_env_ready() -> Path:
    return ddsp_python_executable()
