# README
# Phillip Long
# January 14, 2025

# Synthesize data as audio.

# python /home/pnlong/synthesize_audio/synthesize.py
# python synthesize.py --dataset_filepath "/deepfreeze/pnlong/PDMX/PDMX.csv" --output_dir "/deepfreeze/pnlong" --jobs 10 --soundfont_filepath "/home/pnlong/synthesize_audio/soundfonts/SGM-V2.01.sf2"

# IMPORTS
##################################################

from safetensors.torch import save_file

import argparse
from os.path import exists, dirname, expanduser
from os import mkdir, makedirs, remove
import pandas as pd
import torch
import numpy as np
import tempfile
import subprocess
import multiprocessing
from tqdm import tqdm

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from model_musescore import load, MusicRender

##################################################


# CONSTANTS
##################################################

# some filepaths
PDMX_FILEPATH = "/deepfreeze/pnlong/PDMX/PDMX/PDMX.csv"
OUTPUT_DIR = "/deepfreeze/pnlong/PDMX"
SOUNDFONT_PATH = "/data3/pnlong/musescore/soundfonts/airfont_380_final.sf2"

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

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Synthesize", description = "Synthesize PDMX as audio waveforms stored in tensors.")
    parser.add_argument("-df", "--dataset_filepath", default = PDMX_FILEPATH, type = str, help = "Filepath to full dataset")
    parser.add_argument("-o", "--output_dir", default = OUTPUT_DIR, type = str, help = "Output directory")
    parser.add_argument("-sf", "--soundfont_filepath", default = SOUNDFONT_PATH, type = str, help = "Filepath to soundfont")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Whether or not to recreate files")
    parser.add_argument("-rt", "--reset_tables", action = "store_true", help = "Whether or not to reset data table(s) without recreating files")
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

    # filepaths
    output_dir = f"{args.output_dir}/{DATASET_NAME}"
    if not exists(output_dir) or args.reset:
        makedirs(output_dir, exist_ok = True)
    output_filepath = f"{output_dir}/data.csv"
    stems_output_filepath = f"{output_dir}/stems.csv"
    data_dir = f"{output_dir}/data"
    if not exists(data_dir) or args.reset:
        mkdir(data_dir)

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
    dataset = dataset.drop(columns = ["metadata", "mxl", "pdf", "version"]) # we don't care about metadata
    del convert_to_absolute_path

    # output paths
    dataset["path_output"] = list(map(lambda path: output_dir + ".".join(path[len(original_dataset_dir):].split(".")[:-1]) + ".safetensors", dataset["path"]))

    # filter dataset if desired
    # dataset = dataset[dataset["subset:no_license_conflict"]]
    dataset = dataset.reset_index(drop = True) # reset indicies, as we do dataset based indexing later

    ##################################################


    # WRITE TO FILES
    ##################################################

    # create necessary directory trees if required
    data_subdirectories = set(map(dirname, dataset["path_output"]))
    for data_subdirectory in data_subdirectories:
        makedirs(data_subdirectory, exist_ok = True)
    del data_subdirectories # free up memory

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


    # HELPER FUNCTION TO GET WAVEFORM GIVEN A MUSICRENDER OBJECT
    ##################################################

    # helper function to get the waveform of a given music object
    def get_waveform_tensor(music: MusicRender, temp_dir: tempfile.TemporaryDirectory) -> torch.tensor:
        """
        Given a music object, return it's waveform.
        """

        # write temporary midi file
        midi_path = f"{temp_dir}/temp.mid"
        music.write(path = midi_path) # write the MusicRender object to a temporary .mid file

        # synthesize music as audio
        result = subprocess.run(
            args = [
                "fluidsynth",
                "-T", "raw",
                "-F-",
                "-r", str(SAMPLE_RATE),
                "-g", str(GAIN),
                "-i", args.soundfont_filepath,
                midi_path,
            ],
            check = True, stdout = subprocess.PIPE, stderr = subprocess.DEVNULL,
        ) # synthesize the .mid file using fluidsynth

        # remove midi file
        remove(midi_path)

        # decode bytes to waveform
        waveform = np.frombuffer(result.stdout, dtype = np.int16).reshape(-1, 2).transpose() # transpose so channels are first dimension, time is second dimension
        waveform = torch.from_numpy(waveform.copy())

        # return the stereo waveform
        return waveform

    ##################################################


    # HELPER FUNCTION TO WRITE THE SAFETENSOR CONTAINING ALL STEMS OF A GIVEN MUSICRENDER OBJECT
    ##################################################

    # helper function to convert a music object into a safetensor
    def write_safetensor(music: MusicRender, path_output: str):
        """
        Given a MusicRender object, write the desired 
        information to a safetensor file at path_output.
        """

        # list of waveforms
        waveforms = [None] * len(music.tracks)

        # keep track of longest tensor (most samples)
        max_waveform_length = -1

        # synthesize 
        with tempfile.TemporaryDirectory() as temp_dir: # create a temporary directory

            # loop through stems
            for i in range(len(music.tracks)):

                # get single-track music object
                track = music.tracks[i]
                if len(track.notes) > MAX_N_NOTES_IN_STEM:
                    track.notes = track.notes[:MAX_N_NOTES_IN_STEM]

                # get waveform for this stem
                waveform = get_waveform_tensor(
                    music = MusicRender(
                        metadata = music.metadata,
                        resolution = music.resolution,
                        tempos = music.tempos,
                        key_signatures = music.key_signatures,
                        time_signatures = music.time_signatures,
                        barlines = music.barlines,
                        beats = music.beats,
                        lyrics = music.lyrics,
                        annotations = music.annotations,
                        tracks = [track],
                        song_length = music.song_length,
                        infer_velocity = music.infer_velocity,
                        absolute_time = music.absolute_time,
                    ), 
                    temp_dir = temp_dir)
                
                # update max waveform length
                if waveform.shape[-1] > max_waveform_length:
                    max_waveform_length = waveform.shape[-1]

                # add waveform to stems tensors dictionary
                waveforms[i] = waveform

                # free up memory
                del waveform

        # transform waveforms
        for i in range(len(waveforms)):

            # zero pad the end so all stems are the same length
            waveforms[i] = torch.nn.functional.pad(
                input = waveforms[i], 
                pad = (0, max_waveform_length - waveforms[i].shape[-1]), 
                mode = "constant", 
                value = 0,
            )

            # peak normalize
            waveforms[i] = waveforms[i].type(torch.float) / torch.max(input = waveforms[i])

        # save safetensor to output path
        tensors = {str(i): waveform for i, waveform in enumerate(waveforms)} # keys need to be strings in safetensors
        del waveforms, max_waveform_length # free up memory
        save_file(tensors = tensors, filename = path_output) # save safetensor
        del tensors # free up memory
        
    ##################################################


    # USE MULTIPROCESSING TO ITERATE OVER ALL SONGS IN DATASET
    ##################################################
            
    # helper function to save safetensors given an index
    def synthesize_song_at_index(i: int):
        """
        Given the dataset index, read the song as a music object, and 
        generate a safetensor that contains the audio waveforms for each stem.
        Writes to files.
        """

        # get path output
        path_output = dataset.at[i, "path_output"]

        # save safetensor if necessary
        loaded_music = False # track whether the music object has been loaded
        create_safetensor = not exists(path_output) or args.reset
        if create_safetensor:

            # load music object
            music = load(path = dataset.at[i, "path"])
            loaded_music = True

            # save waveforms as file
            write_safetensor(music = music, path_output = path_output)

        # write tables info to file
        if create_safetensor or args.reset_tables:

            # load music object if necessary
            if not loaded_music:
                music = load(path = dataset.at[i, "path"])

            # write stems info to file
            stems_info = pd.DataFrame(
                data = [(
                    path_output, # path at which this stem can be found
                    j, # index of stem in safetensor
                    track.program, # MIDI program of stem
                    track.is_drum, # whether the track is a drum
                    " ".join(track.name.replace(",", " ").split()) if track.name is not None else None, # name of track, cleaning up the string a bit
                    len(track.annotations) > 0, # whether the track has annotations
                    len(track.lyrics) > 0, # whether the track has lyrics
                    )
                    for j, track in enumerate(music.tracks)],
                columns = STEMS_TABLE_COLUMNS,
            )
            stems_info.to_csv(
                path_or_buf = stems_output_filepath, sep = ",", na_rep = NA_STRING, header = False, index = False, mode = "a",
            )
            del stems_info

            # write line of song info to file
            song_info = dataset.loc[i].to_dict() # get row of dataset as dictionary
            song_info["path"] = path_output # set path to the new output path
            del song_info["path_output"]
            song_info = pd.DataFrame(
                data = [song_info],
                columns = SONGS_TABLE_COLUMNS,
            )
            song_info.to_csv(
                path_or_buf = output_filepath, sep = ",", na_rep = NA_STRING, header = False, index = False, mode = "a",
            )
            del song_info

    ##################################################


    # USE MULTIPROCESSING TO CALL FUNCTIONS
    ##################################################

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(
            iterable = pool.imap_unordered(
                func = synthesize_song_at_index,
                iterable = dataset.index,
                chunksize = CHUNK_SIZE,
            ),
            desc = "Generating Stems",
            total = len(dataset),
        ))

    ##################################################
    
##################################################