# README
# Phillip Long
# October 4, 2024

# Test different soundfonts.

# python /home/pnlong/synthesize_audio/test_soundfonts.py

# IMPORTS
##################################################

import argparse
import logging
from os.path import basename, exists
from os import listdir, mkdir
import multiprocessing
from tqdm import tqdm

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from model_musescore import load, read_musescore

##################################################


# CONSTANTS
##################################################

MUSIC_FILEPATH = "/deepfreeze/pnlong/PDMX_experiments/test_data/debussy/clair_de_lune.mscz"
SOUNDFONTS_DIR = "/deepfreeze/pnlong/soundfonts"

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Test Soundfonts", description = "Test different soundfonts.")
    parser.add_argument("-p", "--path", default = MUSIC_FILEPATH, type = str, help = "Music path for which we want to generate audio")
    parser.add_argument("-s", "--soundfonts_dir", default = SOUNDFONTS_DIR, type = str, help = "Path of the soundfont")
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

    ##################################################

    
    # GET SOUNDFONT PATHS, LOAD MUSIC OBJECT
    ##################################################

    # infer output path if not already provided
    soundfont_paths = list(filter(lambda soundfont_basename: any(soundfont_basename.lower().endswith(filetype) for filetype in ("sf2", "sf3")), listdir(args.soundfonts_dir)))
    soundfont_paths = list(map(lambda soundfont_basename: f"{args.soundfonts_dir}/{soundfont_basename}", soundfont_paths))
    soundfont_paths.append(None) # for the default soundfont

    # load in music object
    if args.path.endswith("mscz"):
        music = read_musescore(path = args.path)
    elif args.path.endswith("json"):
        music = load(path = args.path)
    else:
        raise RuntimeError(f"Invalid path {args.path}. Path must be of type `mscz` or `json`.")
    
    # determine output directory
    output_dir = f"{args.soundfonts_dir}/{basename(args.path).split('.')[0]}"
    if not exists(output_dir):
        mkdir(output_dir)

    ##################################################


    # GENERATE SAMPLE AUDIO FILES
    ##################################################

    # helper function
    def write_audio_from_soundfont(soundfont_path: str):
        """Generate audio sample for a given soundfont."""
        output_path = basename(soundfont_path).split(".")[0] if soundfont_path is not None else "default"
        # logging.info(f"{output_path}:")
        output_path = f"{output_dir}/{output_path}.wav"
        if not exists(output_path) or args.reset:
            music.write(path = output_path, soundfont_path = soundfont_path)

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(pool.map(func = write_audio_from_soundfont, iterable = tqdm(iterable = soundfont_paths, desc = f"Testing Different Soundfonts", total = len(soundfont_paths)), chunksize = 1))
    logging.info("Done!")

    ##################################################

##################################################
