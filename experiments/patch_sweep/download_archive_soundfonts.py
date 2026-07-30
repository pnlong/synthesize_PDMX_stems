"""Download SF2 files from the Internet Archive free-soundfonts-sf2-2019-04 collection."""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

from shared.config import SOUNDFONT_DIR

ARCHIVE_ITEM = "free-soundfonts-sf2-2019-04"
ARCHIVE_BASE = f"https://archive.org/download/{ARCHIVE_ITEM}"
DEFAULT_DEST = Path(SOUNDFONT_DIR) / "archive-2019-04"
CATALOG_PATH = Path(__file__).resolve().parent / "archive_soundfonts.yaml"

PIANO_HINTS = (
    "piano", "grand", "cadenza", "symphony_hall", "nicepiano", "musescore",
    "merlin_grand", "chorium", "saphyr",
)
CHOIR_HINTS = ("acapella", "choir", "chorium", "vocal")
ORCHESTRA_HINTS = ("orchestra", "symphony", "fluidr3", "arachno", "merlin_orchestra", "merlin_symphony")


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(
        description="Download SF2 soundfonts from archive.org free-soundfonts-sf2-2019-04.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_DEST),
        type=str,
        help=f"Download directory (default: {DEFAULT_DEST}).",
    )
    parser.add_argument(
        "--priority-only",
        action="store_true",
        help="Download only piano/orchestra-tagged candidates first (~25 banks).",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Optional cap on number of files to download.",
    )
    parser.add_argument(
        "--write-catalog",
        action="store_true",
        default=True,
        help="Write experiments/patch_sweep/archive_soundfonts.yaml (default: on).",
    )
    parser.add_argument(
        "--no-write-catalog",
        action="store_true",
        help="Skip writing archive_soundfonts.yaml.",
    )
    return parser.parse_args(args=args, namespace=namespace)


def fetch_metadata() -> list[dict]:
    url = f"https://archive.org/metadata/{ARCHIVE_ITEM}"
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(response)
    files = []
    for entry in payload.get("files", []):
        name = entry.get("name", "")
        if not name.lower().endswith(".sf2"):
            continue
        files.append({
            "name": name,
            "size": int(entry.get("size", 0)),
            "url": f"{ARCHIVE_BASE}/{urllib.parse.quote(name)}",
        })
    return sorted(files, key=lambda row: row["name"].lower())


def slugify(name: str) -> str:
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return slug or "soundfont"


def tag_soundfont(name: str) -> list[str]:
    lower = name.lower()
    tags: list[str] = []
    if any(hint in lower for hint in PIANO_HINTS):
        tags.append("piano")
    if any(hint in lower for hint in CHOIR_HINTS):
        tags.append("choir")
    if any(hint in lower for hint in ORCHESTRA_HINTS):
        tags.append("orchestra")
    if "drum" in lower or "perc" in lower:
        tags.append("drums")
    if not tags:
        tags.append("general")
    return tags


def should_download(entry: dict, *, priority_only: bool) -> bool:
    if not priority_only:
        return True
    tags = tag_soundfont(entry["name"])
    return "piano" in tags or "orchestra" in tags or "choir" in tags


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "spdmx-soundfont-downloader/1.0"})
    with urllib.request.urlopen(request, timeout=600) as response:
        data = response.read()
    tmp.write_bytes(data)
    tmp.replace(dest)


def write_catalog(entries: list[dict], dest_dir: Path) -> None:
    candidates = []
    for entry in entries:
        rel = Path("archive-2019-04") / entry["name"]
        tags = tag_soundfont(entry["name"])
        candidates.append({
            "id": slugify(entry["name"]),
            "file": str(rel).replace("\\", "/"),
            "archive_name": entry["name"],
            "size_mb": round(entry["size"] / 1_000_000, 1),
            "source": ARCHIVE_BASE,
            "tags": tags,
            "phase1_status": "archive_candidate",
        })
    catalog = {
        "collection": ARCHIVE_ITEM,
        "archive_url": f"https://archive.org/download/{ARCHIVE_ITEM}",
        "local_dir": str(dest_dir),
        "candidates": candidates,
    }
    with open(CATALOG_PATH, "w") as f:
        yaml.safe_dump(catalog, f, sort_keys=False, allow_unicode=True)


def main():
    args = parse_args()
    if args.no_write_catalog:
        write_catalog_flag = False
    else:
        write_catalog_flag = args.write_catalog

    dest_dir = Path(args.output_dir)
    metadata = fetch_metadata()
    selected = [entry for entry in metadata if should_download(entry, priority_only=args.priority_only)]
    if args.limit is not None:
        selected = selected[: args.limit]

    total_bytes = sum(entry["size"] for entry in selected)
    print(f"Archive: {ARCHIVE_BASE}")
    print(f"Destination: {dest_dir}")
    print(f"Files to fetch: {len(selected)} ({total_bytes / 1e9:.2f} GiB)")

    downloaded = 0
    skipped = 0
    for idx, entry in enumerate(selected, start=1):
        dest = dest_dir / entry["name"]
        if dest.is_file() and dest.stat().st_size > 0:
            skipped += 1
            continue
        print(f"[{idx}/{len(selected)}] {entry['name']} ({entry['size'] / 1e6:.1f} MB)")
        download_file(entry["url"], dest)
        downloaded += 1

    if write_catalog_flag:
        write_catalog(metadata, dest_dir)
        print(f"Wrote catalog: {CATALOG_PATH}")

    print(f"Done. downloaded={downloaded} skipped={skipped} total_catalog={len(metadata)}")


if __name__ == "__main__":
    main()
