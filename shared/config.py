"""Shared configuration for the sPDMX pipeline."""

PDMX_FILEPATH = "/deepfreeze/pnlong/PDMX/PDMX/PDMX.csv"
OUTPUT_DIR = "/deepfreeze/pnlong/SPDMX"
SOUNDFONT_PATH = "/data3/pnlong/soundfonts/SGM-V2.01.sf2"

CHUNK_SIZE = 1
NA_STRING = "NA"

DATA_DIR_NAME = "data"
STEMS_FILE_NAME = "stems"
CAPTIONS_FILE_NAME = "captions"

# {OUTPUT_DIR}/dev/ — development artifacts (ablations, analysis, interim stems)
DEV_DIR_NAME = "dev"

# {OUTPUT_DIR}/dev/stems/ — full-scale stem synthesis (synthesize.py --full)
STEMS_DIR_NAME = "stems"
STEMS_REALIFY_DIR_NAME = "stems_realify"

# {OUTPUT_DIR}/dev/ablations/{basic,basic_realify,slakh,slakh_realify}/ — listening test sample
ABLATIONS_DIR_NAME = "ablations"

# {OUTPUT_DIR}/dev/analysis/ — analysis outputs (song lengths, etc.)
ANALYSIS_DIR_NAME = "analysis"
SONG_LENGTHS_DIR_NAME = "song_lengths"

# {OUTPUT_DIR}/SPDMX/ — assembled sPDMX dataset (via build_spdmx.py, not implemented yet)
SPDMX_DATASET_DIR_NAME = "SPDMX"

ABLATION_SUBSET_COLUMN = "subset:rated_deduplicated"
ABLATION_SAMPLE_SIZE = 100
ABLATION_SAMPLE_SEED = 42

STEMS_TABLE_COLUMNS = [
    "path", "track", "program", "is_drum", "name", "has_lyrics",
]

SONGS_TABLE_COLUMNS = [
    "path", "is_user_pro", "is_user_publisher", "is_user_staff",
    "has_paywall", "is_rated", "is_official", "is_original", "is_draft",
    "has_custom_audio", "has_custom_video", "n_comments", "n_favorites",
    "n_views", "n_ratings", "rating", "license", "license_url", "license_conflict",
    "genres", "groups", "tags", "song_name", "title", "subtitle", "artist_name",
    "composer_name", "publisher", "complexity", "n_tracks", "tracks",
    "song_length", "song_length.seconds", "song_length.bars", "song_length.beats",
    "n_notes", "notes_per_bar", "n_annotations", "has_annotations", "n_lyrics",
    "has_lyrics", "n_tokens", "pitch_class_entropy", "scale_consistency",
    "groove_consistency", "is_best_path", "is_best_arrangement",
    "is_best_unique_arrangement", "subset:all", "subset:rated",
    "subset:deduplicated", "subset:rated_deduplicated",
    "subset:no_license_conflict", "subset:valid_mxl_pdf",
]

CAPTION_MD_COLUMNS = [
    "genres", "groups", "tags", "song_name", "title", "subtitle",
    "artist_name", "composer_name", "publisher", "complexity", "license",
]

CAPTIONS_TABLE_COLUMNS = ["path", "track", "prompt"]

SAMPLE_RATE = 44100
GAIN = 1.0
STEM_FILE_PATTERN = "stem_{track}.flac"
MIXTURE_FILE_NAME = "mixture.flac"
MIXTURE_PEAK_LIMIT = 1.0
FLAC_SUBTYPE = "PCM_16"  # on-disk stems/mixtures; processing uses float32 internally
DEFAULT_AUDIO_FORMAT = "flac"
PROTOTYPE_AUDIO_FORMAT = "mp3"

TARGET_LOUDNESS_LUFS = -23.0

RENDER_MODE_BASIC = "basic"
RENDER_MODE_SLAKH = "slakh"

MAX_N_NOTES_IN_STEM = 50_000
MAX_STEM_DURATION = 30 * 60
MAX_N_SAMPLES_IN_STEM = int(MAX_STEM_DURATION * SAMPLE_RATE)

SA3_SMALL_MUSIC_MAX_DURATION = 120
SA3_MEDIUM_MAX_DURATION = 380
