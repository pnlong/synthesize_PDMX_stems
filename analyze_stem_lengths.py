# README
# Phillip Long
# March 6, 2024

# Analyze the distribution of stem lengths (in number of notes).

# python /home/pnlong/synthesize_audio/analyze_stem_lengths.py

# IMPORTS
##################################################

import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
import argparse
from typing import List
import logging
from os.path import exists, dirname

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from synthesize_manager import PDMX_FILEPATH, CHUNK_SIZE
from model_musescore import load

##################################################


# CONSTANTS
##################################################

# output filepath, must be a NPY file
OUTPUT_FILEPATH = "/deepfreeze/pnlong/PDMX_experiments/analyses/stem_lengths.npy"

# quantiles to analyze
QUANTILES = [0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 0.975, 0.98, 0.985, 0.99, 0.9925, 0.995, 0.9975, 0.999]

##################################################


# HELPER FUNCTION
##################################################

def get_stem_lengths(path: str) -> List[int]:
    """
    Helper function that, given the path to a MusicRender JSON file, 
    returns a list of the lengths of each track (in notes).
    """
    music = load(path = path) # load in music object
    return [len(track.notes) for track in music.tracks] # return length (in notes) of each track

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Analyze Stem Lengths", description = "Analyze stem lengths (in number of notes) of stems in PDMX.")
    parser.add_argument("-df", "--dataset_filepath", default = PDMX_FILEPATH, type = str, help = "Filepath to full PDMX dataset")
    parser.add_argument("-of", "--output_filepath", default = OUTPUT_FILEPATH, type = str, help = "Filepath of output NPY file of stem lengths")
    parser.add_argument("-j", "--jobs", default = int(multiprocessing.cpu_count() / 4), type = int, help = "Number of jobs")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate files")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SET UP
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # read in PDMX
    dataset = pd.read_csv(filepath_or_buffer = args.dataset_filepath, sep = ",", header = 0, index_col = False)
    original_dataset_dir = dirname(args.dataset_filepath)
    dataset["path"] = list(map(lambda path: original_dataset_dir + path[1:], dataset["path"])) # convert path to absolute path
    del original_dataset_dir

    ################################################## 


    # PARSE STEM LENGTHS
    ##################################################

    # avoid calculations if possible
    if not exists(args.output_filepath) or args.reset:

        # use multiprocessing
        with multiprocessing.Pool(processes = args.jobs) as pool:
            results = list(pool.map(func = get_stem_lengths,
                                    iterable = tqdm(iterable = dataset["path"],
                                                   desc = "Parsing Stem Lengths",
                                                   total = len(dataset)),
                                    chunksize = CHUNK_SIZE))
        
        # wrangle results and save
        results = sum(results, []) # flatten results
        results = np.array(sorted(results)) # sort results
        np.save(file = args.output_filepath, arr = results) # save as numpy object
        del results # free up memory

    ##################################################


    # ANALYZE STEM LENGTHS
    ##################################################

    # load in results
    results = np.load(file = args.output_filepath)

    # output quantiles
    logging.info("Quantiles:")
    for quantile, quantile_value in zip(QUANTILES, np.quantile(results, q = QUANTILES)):
        logging.info(f"- {100 * quantile:.2f}%: {quantile_value:.4f}")

    ##################################################

##################################################