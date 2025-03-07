# README
# Phillip Long
# October 4, 2024

# Synthesize the rated and deduplicated subset of PDMX as audio.

# python /home/pnlong/synthesize_audio/synthesize_rd_subset.py

# IMPORTS
##################################################

import argparse
import logging
from os.path import exists, dirname, expanduser
from os import mkdir, makedirs
import subprocess
import multiprocessing
from tqdm import tqdm
import pandas as pd
# from shutil import copy
import tempfile
import subprocess
# from typing import Tuple

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from synthesize_helper import SAMPLE_RATE, GAIN
from synthesize import PDMX_FILEPATH, OUTPUT_DIR, SOUNDFONT_PATH, NA_STRING
from model_musescore import load

##################################################


# CONSTANTS
##################################################

# name of dataset
DATASET_NAME = "sPDMX"

# filetype to synthesize
AUDIO_FILETYPE_TO_SYNTHESIZE = "wav"

# chunk size for multiprocessing
CHUNK_SIZE = 1

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Synthesize Rated and Deduplicated Subset", description = "Synthesize audio for the rated and deduplicated subset of PDMX (as a testbed).")
    parser.add_argument("-df", "--dataset_filepath", default = PDMX_FILEPATH, type = str, help = "Filepath to full dataset")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory")
    parser.add_argument("-sf", "--soundfont_filepath", default = SOUNDFONT_PATH, type = str, help = "Filepath to soundfont")
    # parser.add_argument("-em", "--exclude_metadata", action = "store_true", help = "Whether to copy over metadata")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate files")
    parser.add_argument("-j", "--jobs", default = int(multiprocessing.cpu_count() / 4), type = int, help = "Number of jobs")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # set up logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # filepaths
    output_dir = f"{args.output_dir}/{DATASET_NAME}"
    if not exists(output_dir):
        makedirs(output_dir)
    output_filepath = f"{output_dir}/{DATASET_NAME}.csv"
    data_dir = f"{output_dir}/data"
    if not exists(data_dir):
        mkdir(data_dir)
    # metadata_dir = f"{output_dir}/metadata"
    # if not args.exclude_metadata and not exists(metadata_dir):
    #     mkdir(metadata_dir)
    # facets_dir = f"{output_dir}/subset_paths"
    # if not exists(facets_dir):
    #     mkdir(facets_dir)

    # soundfont
    if args.soundfont_filepath is None:
        args.soundfont_filepath = f"{expanduser('~')}/.muspy/musescore-general/MuseScore_General.sf3" # musescore soundfont path
    if not exists(args.soundfont_filepath):
        raise RuntimeError("Soundfont not found. Please download it by `muspy.download_musescore_soundfont()`.")

    # load in dataset
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)
    original_dataset_dir = dirname(args.dataset_filepath)
    convert_to_absolute_path = lambda path: original_dataset_dir + path[1:]
    dataset["path"] = list(map(convert_to_absolute_path, dataset["path"])) # convert path to absolute path
    # dataset["metadata"] = list(map(convert_to_absolute_path, dataset["metadata"])) # convert metadata to absolute path
    dataset["path_output"] = list(map(lambda path: output_dir + ".".join(path[len(original_dataset_dir):].split(".")[:-1]) + "." + AUDIO_FILETYPE_TO_SYNTHESIZE, dataset["path"]))
    # dataset["metadata_output"] = list(map(lambda path: (output_dir + path[len(original_dataset_dir):]) if not pd.isna(path) else None, dataset["metadata"]))
    del original_dataset_dir, convert_to_absolute_path

    # create necessary directory trees if required
    data_subdirectories = set(map(dirname, dataset["path_output"]))
    for data_subdirectory in data_subdirectories:
        makedirs(data_subdirectory, exist_ok = True)
    del data_subdirectories # free up memory
    # if not args.exclude_metadata:
    #     metadata_subdirectories = set(map(dirname, filter(lambda path: not pd.isna(path), dataset["metadata_output"])))
    #     for metadata_subdirectory in metadata_subdirectories:
    #         makedirs(metadata_subdirectory, exist_ok = True)
    #     del metadata_subdirectories # free up memory

    ##################################################


    # FILTER DATASET IF NECESSARY
    ##################################################

    # only want the rated and deduplicated subset for now
    dataset = dataset[dataset["subset:rated_deduplicated"]]

    # reset indicies, as we do dataset based indexing later
    dataset = dataset.reset_index(drop = True)

    ##################################################


    # SYNTHESIZE FILES
    ##################################################

    # helper function to save files
    # def synthesize_song_at_index(i: int) -> Tuple[str, str]:
    def synthesize_song_at_index(i: int) -> str:
        """
        Given the dataset index, read the song as a music object, and write as audio.
        Copy metadata if applicable.
        Returns the output path.
        """

        # save as music object
        path_output = dataset.at[i, "path_output"]
        if not exists(path_output) or args.reset:
            music = load(path = dataset.at[i, "path"])
            with tempfile.TemporaryDirectory() as temp_dir: # create a temporary directory
                midi_path = f"{temp_dir}/temp.mid"
                music.write(path = midi_path) # write the MusicRender object to a temporary .mid file
                subprocess.run(
                    args = ["fluidsynth", "-ni", "-F", path_output, "-T", "auto", "-r", str(SAMPLE_RATE), "-g", str(GAIN), args.soundfont_filepath, midi_path],
                    check = True, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL,
                ) # synthesize the .mid file using fluidsynth

        # return output path
        path_output = "." + path_output[len(output_dir):]
        return path_output

        # # copy over metadata path
        # metadata_path = dataset.at[i, "metadata"]
        # metadata_path_output = dataset.at[i, "metadata_output"]
        # if not args.exclude_metadata and metadata_path is not None and (not exists(metadata_path_output) or args.reset):
        #     copy(src = metadata_path, dst = metadata_path_output) # copy over metadata
        # if metadata_path_output is not None:
        #     metadata_path_output = "." + metadata_path_output[len(output_dir):]

        # # return paths
        # return (path_output, metadata_path_output)

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        # dataset["path"], dataset["metadata"] = list(zip(*list(
        #     pool.map(func = synthesize_song_at_index,
        #              iterable = tqdm(iterable = dataset.index, desc = f"Generating {DATASET_NAME}", total = len(dataset)),
        #              chunksize = CHUNK_SIZE)
        # )))
        dataset["path"] = list(pool.map(
            func = synthesize_song_at_index,
            iterable = tqdm(iterable = dataset.index, desc = f"Generating {DATASET_NAME}", total = len(dataset)),
            chunksize = CHUNK_SIZE))

    ##################################################


    # SORT OUT DATASET QUIRKS
    ##################################################

    # remove unnecessary columns
    # dataset = dataset.drop(columns = ["path_output", "metadata_output"] + (["metadata"] if args.exclude_metadata else []))
    dataset = dataset.drop(columns = ["path_output", "metadata"]) 
    dataset.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_STRING, header = True, index = False, mode = "w")

    # # text files with paths for each facet
    # for column in list(filter(lambda column: column.startswith("subset:"), dataset.columns)):
    #     with open(f"{facets_dir}/{column.split(':')[-1]}.txt", "w") as output_file:
    #         output_file.write("\n".join(dataset[dataset[column]]["path"]))

    ##################################################

##################################################