# README
# Phillip Long
# March 20, 2025

# Check synthesized data and make sure there are no corrupt/incomplete files.

# python /home/pnlong/synthesize_audio/test_synthesis.py
# python test_synthesis.py --dataset_dir "/deepfreeze/user_shares/pnlong/sPDMX_stems" --output_filepath "/home/pnlong/synthesize_audio/test.csv"

# IMPORTS
##################################################

from safetensors import safe_open

import argparse
from os.path import exists, dirname
import logging
import pandas as pd
import multiprocessing
from tqdm import tqdm

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

from synthesize import OUTPUT_DIR, DATASET_NAME, STEMS_FILE_NAME, NA_STRING, CHUNK_SIZE

##################################################


# CONSTANTS
##################################################

# filepaths
DATASET_DIR = f"{OUTPUT_DIR}/{DATASET_NAME}"
OUTPUT_FILEPATH = f"{DATASET_DIR}/test.csv"

# output columns
OUTPUT_COLUMNS = ["path", "valid", "complete", "n_missing_stems"]

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Test", description = "Test synthesized safetensors.")
    parser.add_argument("-d", "--dataset_dir", default = DATASET_DIR, help = "Directory of dataset")
    parser.add_argument("-o", "--output_filepath", default = OUTPUT_FILEPATH, type = str, help = "Output filepath with test results")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to retest files")
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

    # load in data frame
    dataset = pd.read_csv(filepath_or_buffer = f"{args.dataset_dir}/{STEMS_FILE_NAME}.csv", sep = ",", header = 0, index_col = False, usecols = ["path", "track"])
    dataset = dataset.groupby(by = "path").size()

    # set up logging
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # write output columns
    if not exists(args.output_filepath) or args.reset:
        pd.DataFrame(columns = OUTPUT_COLUMNS).to_csv(
            path_or_buf = args.output_filepath, sep = ",", na_rep = NA_STRING, header = True, index = False, mode = "w",
        )
    completed_paths = set(pd.read_csv(filepath_or_buffer = args.output_filepath, sep = ",", header = 0, index_col = False, usecols = ["path"])["path"])

    ##################################################


    # GO THROUGH FILES AND ENSURE THEY ARE VALID
    ##################################################

    # helper function to test the given path
    def test(path: str):
        """Test the given path to ensure completeness."""
        expected_size = dataset[path] # expected number of tracks
        if path not in completed_paths:
            try: # try opening
                with safe_open(filename = path, framework = "pt", device = "cpu") as file:
                    valid = True
                    actual_size = len(file.keys())
                    complete = (actual_size == expected_size)
            except: # if the file doesn't open it's invalid
                valid = False
                complete = False
                actual_size = 0 # the actual size is 0
            n_missing_stems = expected_size - actual_size
            pd.DataFrame(data = [dict(zip(OUTPUT_COLUMNS, (path, valid, complete, n_missing_stems)))]).to_csv(
                path_or_buf = args.output_filepath, sep = ",", na_rep = NA_STRING, header = False, index = False, mode = "a",
            ) # write to file
            completed_paths.add(path)

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(
            iterable = pool.imap_unordered(
                func = test,
                iterable = dataset.index,
                chunksize = CHUNK_SIZE,
            ),
            desc = "Testing",
            total = len(dataset),
        ))

    ##################################################


    # ANALYZE
    ##################################################

    # load in results
    results = pd.read_csv(filepath_or_buffer = args.output_filepath, sep = ",", header = 0, index_col = False)
    
    # calculate statistics
    n_valid = sum(results["valid"])
    n_invalid = len(results) - n_valid
    n_complete = sum(results["complete"])
    n_incomplete = len(results) - n_complete
    n_missing_values = sorted(pd.unique(results["n_missing_stems"]).tolist())

    # log statistics
    logging.info(f"{len(results)} total songs.")
    get_statistic_string = lambda statistic, n: f"{statistic.title()}: {100 * (n / len(results)):.4f}% ({n:,} songs)."
    for statistic, n in zip(("Valid", "Invalid", "Complete", "Incomplete"), (n_valid, n_invalid, n_complete, n_incomplete)):
        logging.info(get_statistic_string(statistic = statistic, n = n))
    if not (len(n_missing_values) == 1 and n_missing_values[0] == 0): # print missing values if there are any
        logging.info(f"Missing: {n_missing_values}")

    ##################################################
    
##################################################