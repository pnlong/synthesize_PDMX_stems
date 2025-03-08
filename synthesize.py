# README
# Phillip Long
# January 14, 2025

# Synthesize data as audio.

# python /home/pnlong/synthesize_audio/synthesize.py
# python synthesize.py --dataset_filepath "/deepfreeze/pnlong/PDMX/PDMX/PDMX.csv" --output_dir "/deepfreeze/pnlong/PDMX" --jobs 10 --soundfont_filepath "/data3/pnlong/soundfonts/SGM-V2.01.sf2"

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
import mido
from typing import List

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from model_musescore import load

##################################################


# CONSTANTS
##################################################

# some filepaths
PDMX_FILEPATH = "/deepfreeze/pnlong/PDMX/PDMX/PDMX.csv"
OUTPUT_DIR = "/deepfreeze/pnlong/PDMX"
SOUNDFONT_PATH = "/data3/pnlong/soundfonts/SGM-V2.01.sf2"

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
    dataset = dataset[dataset["subset:all_valid"]].reset_index(drop = True) # get only rows with valid midi files
    dataset = dataset.drop(columns = ["metadata", "mxl", "pdf", "version", "subset:all_valid"]) # we don't care about metadata, musicxml, pdf, or version
    original_dataset_dir = dirname(args.dataset_filepath)
    convert_to_absolute_path = lambda path: original_dataset_dir + path[1:]
    dataset["path"] = list(map(convert_to_absolute_path, dataset["path"])) # convert json file path to absolute path
    dataset["mid"] = list(map(convert_to_absolute_path, dataset["mid"])) # convert midi file path to absolute path
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
    completed_paths = set(pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False, usecols = ["path"])["path"])
    if not exists(stems_output_filepath) or args.reset or args.reset_tables: # stem-level dataset
        pd.DataFrame(columns = STEMS_TABLE_COLUMNS).to_csv(
            path_or_buf = stems_output_filepath, sep = ",", na_rep = NA_STRING, header = True, index = False, mode = "w",
        )

    ##################################################


    # HELPER FUNCTION TO WRITE MIDI FILES FOR EACH TRACK IN A GIVEN FILEPATH
    ##################################################

    def create_midi_track_files(path: str, directory: str) -> List[str]:
        """
        Given a path to a MIDI file, write a file for each track in that file in 
        the provided directory. Returns a list of the absolute filepaths to the
        MIDI files for each track.
        """

        # load in the MIDI file
        midi = mido.MidiFile(filename = path, charset = "utf8")

        # initialize return list
        track_paths = [f"{directory}/{i}.mid" for i in range(len(midi.tracks))] # do not include meta track

        # iterate through tracks
        for i, track in enumerate(midi.tracks): # no metatracks in PDMX MIDI files

            # create new midi file for this track
            track_midi = mido.MidiFile(ticks_per_beat = midi.ticks_per_beat, charset = "utf8")

            # add current track
            track_midi_track = mido.MidiTrack()
            n_notes = 0 # track the number of notes seen so far
            for message in track: # copy the messages from the original track to the new track
                if message.type == "note_on" and message.velocity > 0: # if this is a note on event
                    if n_notes > MAX_N_NOTES_IN_STEM: # truncate stem if there are too many notes
                        break 
                    else: # otherwise, increment the number of notes seen so far
                        n_notes += 1
                track_midi_track.append(message)
            track_midi.tracks.append(track_midi_track)

            # write track midi file to temporary path
            track_midi.save(track_paths[i])

            # free up memory
            del track_midi, track_midi_track

        # free up memory
        del midi

        # return list of track paths
        return track_paths

    ##################################################


    # HELPER FUNCTION TO GET WAVEFORM GIVEN A MIDI FILEPATH
    ##################################################

    # helper function to get the waveform of a given music object
    def get_waveform_tensor(path: str) -> torch.tensor:
        """
        Given a path to a MIDI file, return its waveform.
        """        

        # synthesize music as audio
        result = subprocess.run(
            args = [
                "fluidsynth",
                "-T", "raw",
                "-F-",
                "-r", str(SAMPLE_RATE),
                "-g", str(GAIN),
                "-i", args.soundfont_filepath,
                path,
            ],
            check = True, stdout = subprocess.PIPE, stderr = subprocess.DEVNULL,
        ) # synthesize the MIDI file using fluidsynth

        # decode bytes to waveform
        waveform = np.frombuffer(result.stdout, dtype = np.int16).reshape(-1, 2).transpose() # transpose so channels are first dimension, time is second dimension
        waveform = torch.from_numpy(waveform.copy())

        # return the stereo waveform
        return waveform

    ##################################################


    # HELPER FUNCTION TO WRITE THE SAFETENSOR CONTAINING ALL STEMS OF A GIVEN MUSICRENDER OBJECT
    ##################################################

    # helper function to convert a music object into a safetensor
    def write_safetensor(path: str, path_output: str):
        """
        Given the path to a MIDI file, write the desired 
        information to a safetensor file at path_output.
        """

        # synthesize 
        with tempfile.TemporaryDirectory() as temp_dir: # create a temporary directory
            
            # create a MIDI file for each track in the given path
            track_paths = create_midi_track_files(path = path, directory = temp_dir)

            # list of waveforms
            waveforms = [None] * len(track_paths)

            # keep track of longest tensor (most samples)
            max_waveform_length = -1

            # loop through stems
            for i, track_path in enumerate(track_paths):

                # get waveform for this stem
                waveform = get_waveform_tensor(path = track_path)
                
                # update max waveform length
                if waveform.shape[-1] > max_waveform_length:
                    max_waveform_length = waveform.shape[-1]

                # add waveform to stems tensors dictionary
                waveforms[i] = waveform

                # free up space and memory
                remove(track_path) # remove temporary midi file
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
        if not exists(path_output) or args.reset:
            write_safetensor(path = dataset.at[i, "mid"], path_output = path_output) # write safetensor from midi file

        # write tables info to file
        if path_output not in completed_paths:

            # load music object
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
            del song_info["path_output"], song_info["mid"]
            song_info = pd.DataFrame(
                data = [song_info],
                columns = SONGS_TABLE_COLUMNS,
            )
            song_info.to_csv(
                path_or_buf = output_filepath, sep = ",", na_rep = NA_STRING, header = False, index = False, mode = "a",
            )
            del song_info

            # add output path to completed paths
            completed_paths.add(path_output)

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