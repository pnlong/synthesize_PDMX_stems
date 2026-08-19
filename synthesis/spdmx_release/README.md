# sPDMX

Audio stems and dense MIDI for [PDMX](https://zenodo.org/records/13763756)
(Long et al., ICASSP 2025). Each PDMX song is identified by **`song_id`**,
the hashed layout path without `./data/` or `.json`
(for example `8/44/QmQt11bci266XxFpztpSJmfH1ijHX8zRcmaVe1Y7akYZTZ`).

## Layout

```
.
├── LICENSE
├── README.md
├── SPDMX.csv
├── audio/<song_id>/<track>.flac
└── mid/<song_id>.mid
```

`track` in filenames and in `SPDMX.csv` is the **dense** MIDI track index
after empty PDMX tracks are dropped. `original_track` is the PDMX MIDI track
index.

## Primary key

`song_id` is the song primary key. It joins to PDMX.csv as:

| sPDMX | PDMX.csv |
|-------|----------|
| `song_id` | `path` with `./data/` prefix and `.json` suffix stripped |
| `mid` (`./mid/{song_id}.mid`) | `mid` = `./mid/{song_id}.mid` |
| `path` (`./audio/{song_id}`) | PDMX `path` is metadata JSON; sPDMX `path` is the stem directory |

Row identity in `SPDMX.csv` is `(song_id, track)`.

Columns: `song_id`, `path`, `mid`, `track`, `original_track`, `program`, `is_drum`, `name`.

## Citation

Please cite PDMX and this work if you use sPDMX.

```
@inproceedings{long2024pdmx,
  author={Long, Phillip and Novack, Zachary and Berg-Kirkpatrick, Taylor and McAuley, Julian},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={{PDMX}: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing},
  year={2025},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10890217}
}
```

## License

See `LICENSE` in this directory (CC BY 4.0, with PDMX public-domain scores and
optional Stability AI terms for SA3-realified stems).
