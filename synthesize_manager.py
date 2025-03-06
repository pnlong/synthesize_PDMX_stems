# README
# Phillip Long
# January 14, 2025

# Synthesize data as audio.

# python /home/pnlong/synthesize_audio/synthesize_manager.py
# python synthesize_manager.py --dataset_filepath "/deepfreeze/pnlong/PDMX/PDMX/PDMX.csv" --output_dir "/deepfreeze/user_shares/pnlong" --soundfont_filepath "/home/pnlong/soundfonts/SGM-V2.01.sf2" --temporary_storage_dir "/home/pnlong/temp_stems_storage" --jobs 10 --use_tarball_buffer

# IMPORTS
##################################################

import argparse
from os.path import exists, dirname, expanduser, realpath
from os import mkdir, makedirs
import pandas as pd
import logging
from shutil import rmtree
import pickle

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

##################################################


# CONSTANTS
##################################################

# some filepaths
PDMX_FILEPATH = "/deepfreeze/pnlong/PDMX/PDMX.csv"
OUTPUT_DIR = "/deepfreeze/pnlong"
SOUNDFONT_PATH = "/home/pnlong/soundfonts/airfont_380_final.sf2"
TEMPORARY_STORAGE_DIR = "/home/pnlong/temp_stems_storage" # MUST BE ON LOCAL DRIVES, NOT DEEPFREEZE

# multiprocessing chunk size
CHUNK_SIZE = 1

# NA String for CSV files
NA_STRING = "NA"

# name of dataset
DATASET_NAME = "sPDMX_stems"

# for stems data table
STEMS_TABLE_COLUMNS = [
    "path", "track", "program", "is_drum", "name", "has_annotations", "has_lyrics"
]

# for songs data table
SONGS_TABLE_COLUMNS = [
    "path", "version", "is_user_pro", "is_user_publisher", "is_user_staff", 
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
    "subset:no_license_conflict", "subset:valid_mxl_pdf"
]

# for synthesizing audio
SAMPLE_RATE = 44100
GAIN = 1.0

# truncate overly-long stems to avoid memory errors
MAX_N_NOTES_IN_STEM = 100000

# what to name CSV files in resulting dataset
SONG_LEVEL_FILENAME = "data"
STEM_LEVEL_FILENAME = "stems"

# line to separate outputs
LINE = "".join(("=" for _ in range(150)))

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Synthesize", description = "Manager for synthesizing PDMX as audio waveforms stored in tensors.")
    parser.add_argument("-df", "--dataset_filepath", default = PDMX_FILEPATH, type = str, help = "Filepath to full dataset")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory")
    parser.add_argument("-sf", "--soundfont_filepath", default = SOUNDFONT_PATH, type = str, help = "Filepath to soundfont")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate files")
    parser.add_argument("-rt", "--reset_tables", action = "store_true", help = "Whether or not to reset data table(s) without recreating files")
    parser.add_argument("-ut", "--use_tarball_buffer", action = "store_true", help = "Whether or not to use tarball buffering approach to create data")
    parser.add_argument("-gz", "--gzip_tarballs", action = "store_true", help = "Whether or not to GZIP tarballs if tarball buffering approach is in use")
    parser.add_argument("-t", "--temporary_storage_dir", default = TEMPORARY_STORAGE_DIR, type = str, help = "Temporary storage directory, must be specified if use_tarball_buffer flag is set")
    parser.add_argument("-j", "--jobs", default = 10, type = int, help = "Number of Jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # output data directories
    output_dir = f"{args.output_dir}/{DATASET_NAME}"
    if not exists(output_dir) or args.reset:
        makedirs(output_dir, exist_ok = True)
    data_dir = f"{output_dir}/data"
    if not exists(data_dir) or args.reset:
        mkdir(data_dir)

    # create temporary storage directory
    if args.use_tarball_buffer:
        if exists(args.temporary_storage_dir):
            rmtree(args.temporary_storage_dir, ignore_errors = True)
        makedirs(args.temporary_storage_dir)
    
    # job level song-level output filepaths
    output_filepath = f"{output_dir}/{SONG_LEVEL_FILENAME}.csv"
    job_output_filepaths_dir = f"{output_dir}/.job_{SONG_LEVEL_FILENAME}"
    job_output_filepaths = list(map(lambda i: f"{job_output_filepaths_dir}/{SONG_LEVEL_FILENAME}.{i}.csv", range(args.jobs)))
    if not exists(job_output_filepaths_dir) or args.reset or args.reset_tables:
        if exists(job_output_filepaths_dir):
            rmtree(job_output_filepaths_dir)
        mkdir(job_output_filepaths_dir)    
    
    # job level stem-level output filepaths
    stems_output_filepath = f"{output_dir}/{STEM_LEVEL_FILENAME}.csv"
    job_stems_output_filepaths_dir = f"{output_dir}/.job_{STEM_LEVEL_FILENAME}"
    job_stems_output_filepaths = list(map(lambda i: f"{job_stems_output_filepaths_dir}/{STEM_LEVEL_FILENAME}.{i}.csv", range(args.jobs)))
    if not exists(job_stems_output_filepaths_dir) or args.reset or args.reset_tables:
        if exists(job_stems_output_filepaths_dir):
            rmtree(job_stems_output_filepaths_dir)
        mkdir(job_stems_output_filepaths_dir)

    # instructions for jobs
    job_instructions_dir = f"{output_dir}/.job_instructions"
    job_instructions_filepaths = list(map(lambda i: f"{job_instructions_dir}/{i}.pickle", range(args.jobs)))
    if not exists(job_instructions_dir):
        mkdir(job_instructions_dir)

    # soundfont
    if args.soundfont_filepath is None:
        args.soundfont_filepath = f"{expanduser('~')}/.muspy/musescore-general/MuseScore_General.sf3" # musescore soundfont path
    if not exists(args.soundfont_filepath):
        raise FileNotFoundError("Soundfont not found. Please download it by `muspy.download_musescore_soundfont()`.")
    
    ##################################################
    

    # LOAD IN DATASET
    ##################################################

    # load in dataset
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)
    original_dataset_dir = dirname(args.dataset_filepath)
    convert_to_absolute_path = lambda path: original_dataset_dir + path[1:]
    dataset["path"] = list(map(convert_to_absolute_path, dataset["path"])) # convert path to absolute path
    dataset = dataset.drop(columns = ["metadata", "mxl", "pdf", "version"]) # we don't care about metadata
    del convert_to_absolute_path

    # output paths
    dataset["path_output"] = list(map(lambda path: output_dir + ".".join(path[len(original_dataset_dir):].split(".")[:-1]) + ".safetensors", dataset["path"]))

    # filter dataset if desired
    # dataset = dataset[dataset["subset:no_license_conflict"]]
    dataset = dataset.reset_index(drop = True) # reset indicies, as we do dataset based indexing later

    ##################################################


    # DEAL WITH SUBDIRECTORIES
    ##################################################

    # create necessary directory trees if required
    data_subdirectories = sorted(list(set(map(dirname, dataset["path_output"])))) # sort for later
    for data_subdirectory in (map(dirname, data_subdirectories) if args.use_tarball_buffer else data_subdirectories): # only want the parent directories
        makedirs(data_subdirectory, exist_ok = True)

    # divide data subdirectories across different jobs (or get start and end indicies at least)
    start_indicies = list(range(0, len(data_subdirectories), int(len(data_subdirectories) / args.jobs) + 1))
    end_indicies = [start_index for start_index in (start_indicies[1:] + [len(data_subdirectories)])]

    ##################################################


    # WRITE TO FILES
    ##################################################

    # write dataset to temporary file
    dataset_temporary_filepath = f"{output_dir}/.dataset.csv"
    dataset.to_csv(
        path_or_buf = dataset_temporary_filepath, sep = ",", na_rep = NA_STRING, header = True, index = False, mode = "w",
    )

    # write column names
    if not exists(output_filepath) or args.reset or args.reset_tables: # song-level dataset
        pd.DataFrame(columns = SONGS_TABLE_COLUMNS).to_csv(
            path_or_buf = output_filepath, sep = ",", na_rep = NA_STRING, header = True, index = False, mode = "w",
        )
    if not exists(stems_output_filepath) or args.reset or args.reset_tables: # stem-level dataset
        pd.DataFrame(columns = STEMS_TABLE_COLUMNS).to_csv(
            path_or_buf = stems_output_filepath, sep = ",", na_rep = NA_STRING, header = True, index = False, mode = "w",
        )

    ##################################################


    # CALL SYNTHESIZE HELPER PROGRAM
    ##################################################
    
    # software filepaths
    software_dir = dirname(realpath(__file__))
    software_filepath = f"{software_dir}/synthesize_helper.py"

    # easier to read
    logging.info(LINE)

    # run jobs with screen command
    for i, job_instructions_filepath in enumerate(job_instructions_filepaths):
        with open(job_instructions_filepath, "wb") as instructions_file:
            pickle.dump(
                obj = {
                    "dataset_filepath": dataset_temporary_filepath,
                    "output_filepath": job_output_filepaths[i],
                    "stems_output_filepath": job_stems_output_filepaths[i],
                    "subdirectories": data_subdirectories[start_indicies[i]:end_indicies[i]], # list of relevant subdirectories for this job
                    "reset": args.reset,
                    "reset_tables": args.reset_tables,
                    "use_tarball_buffer": args.use_tarball_buffer,
                    "gzip_tarballs": args.gzip_tarballs,
                }, 
                file = instructions_file
            )
        logging.info(
            f"conda activate base; " +
            f"cd {software_dir}; " +
            f"python {software_filepath} --soundfont_filepath {args.soundfont_filepath} --temporary_storage_dir {args.temporary_storage_dir} --instructions_filepath {job_instructions_filepath}"
        ) # output command to call

    # easier to read
    logging.info(LINE)

    ##################################################


    # CLEANUP
    ##################################################

    # concatenate job outputs into megafiles
    logging.info(f"# WRANGLE TEMPORARY OUTPUTS")
    logging.info(f"cat {job_output_filepaths_dir}/* >> {output_filepath} # concatenate song-level dataset") # create song-level dataset
    logging.info(f"cat {job_stems_output_filepaths_dir}/* >> {stems_output_filepath} # concatenate stem-level dataset") # create song-level dataset
    if args.use_tarball_buffer:
        logging.info(f"bash {dirname(realpath(__file__))}/synthesize_helper_extractor.sh {output_dir} # extract tarballs")

    # easier to read
    logging.info(LINE)

    # clear temporary files
    logging.info("# CLEAR TEMPORARY FILES")
    logging.info(f"rm -rf {job_instructions_dir} # remove job instructions")
    logging.info(f"rm -rf {job_output_filepaths_dir} # remove song-level job outputs")
    logging.info(f"rm -rf {job_stems_output_filepaths_dir} # remove stem-level job outputs")
    logging.info(f"rm {dataset_temporary_filepath} # remove temporary dataset")
    if args.use_tarball_buffer:
        logging.info(f"find {data_dir} -name \"*.tar\" | xargs rm # remove tarballs after extraction")

    ##################################################
    
##################################################