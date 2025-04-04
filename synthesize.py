# README
# Phillip Long
# January 14, 2025

# Synthesize data as audio.

# python /home/pnlong/synthesize_audio/synthesize.py
# python synthesize.py --dataset_filepath "/deepfreeze/pnlong/PDMX/PDMX/PDMX.csv" --output_dir "/deepfreeze/user_shares/pnlong" --jobs 10 --soundfont_filepath "/data3/pnlong/soundfonts/SGM-V2.01.sf2"

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

##################################################


# CONSTANTS
##################################################

# some filepaths
PDMX_FILEPATH = "/deepfreeze/pnlong/PDMX/PDMX/PDMX.csv"
OUTPUT_DIR = "/deepfreeze/user_shares/pnlong"
SOUNDFONT_PATH = "/data3/pnlong/soundfonts/SGM-V2.01.sf2"

# multiprocessing chunk size
CHUNK_SIZE = 1

# NA String for CSV files
NA_STRING = "NA"

# name of dataset
DATASET_NAME = "sPDMX_stems"
DATA_DIR_NAME = "data"
STEMS_FILE_NAME = "stems"

# for stems data table
STEMS_TABLE_COLUMNS = [
    "path", "track", "program", "is_drum", "name", "has_lyrics",
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
MAX_N_NOTES_IN_STEM = 50_000 # 0.99997812562 quantile for stem lengths
MAX_STEM_DURATION = 30 * 60 # in seconds
MAX_N_SAMPLES_IN_STEM = int(MAX_STEM_DURATION * SAMPLE_RATE)

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
    output_filepath = f"{output_dir}/{DATA_DIR_NAME}.csv"
    stems_output_filepath = f"{output_dir}/{STEMS_FILE_NAME}.csv"
    data_dir = f"{output_dir}/{DATA_DIR_NAME}"
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


    # SYNTHESIZE STEMS FOR A GIVEN INDEX
    ##################################################

    def synthesize_song_at_index(i: int):
        """
        Given the dataset index, read the song as a music object, and 
        generate a safetensor that contains the audio waveforms for each stem.
        Writes to files.
        """

        # get path output
        path_output = dataset.at[i, "path_output"]

        # save safetensor if necessary
        if path_output not in completed_paths:

            # READ IN MIDI FILE AND CREATE STEM MIDI FILES
            ##################################################

            # load in the MIDI file
            midi = mido.MidiFile(filename = dataset.at[i, "mid"], charset = "utf8")

            # determine whether we are synthesizing (or the song has already been synthesized)
            need_to_synthesize = not exists(path_output) or args.reset            

            # initialize paths for each stem MIDI file list
            if need_to_synthesize:
                temp_dir = tempfile.TemporaryDirectory() # create temporary directory
                track_paths = [f"{temp_dir.name}/{i}.mid" for i in range(len(midi.tracks))] # track paths in temporary directory

            # iterate through tracks
            for j, track in enumerate(midi.tracks): # no metatracks in PDMX MIDI files

                # create new midi file for this track
                if need_to_synthesize:
                    track_midi = mido.MidiFile(ticks_per_beat = midi.ticks_per_beat, charset = "utf8")
                    track_midi_track = mido.MidiTrack() # add current track

                # set defaults
                program = 0
                is_drum = False
                track_name = None
                has_lyrics = False

                # iterate through track and parse information
                n_notes = 0 # track the number of notes seen so far
                determined_whether_track_is_drum = False
                for message in track: # copy the messages from the original track to the new track
                    if message.type == "note_on" and message.velocity > 0: # if this is a note on event
                        n_notes += 1
                    elif message.type == "program_change":
                        program = message.program
                    elif message.type == "track_name":
                        track_name = message.name
                        track_name = " ".join(track_name.replace(",", " ").split()) # clean up track name
                    elif message.type == "lyrics":
                        has_lyrics = True
                    if not determined_whether_track_is_drum and hasattr(message, "channel"):
                        is_drum = (message.channel == 9)
                        determined_whether_track_is_drum = True
                    if need_to_synthesize and n_notes <= MAX_N_NOTES_IN_STEM: # if we are synthesizing, add to the track midi track
                        track_midi_track.append(message)
                del n_notes, determined_whether_track_is_drum # free up memory

                # write track midi file to temporary path
                if need_to_synthesize:
                    track_midi.tracks.append(track_midi_track)
                    track_midi.save(track_paths[j])
                    del track_midi, track_midi_track # free up memory

                # write stem info
                pd.DataFrame( # create dataframe with one row
                    data = [dict(zip(STEMS_TABLE_COLUMNS, (
                        path_output, # path at which this stem can be found
                        j, # index of stem in safetensor
                        program, # MIDI program of stem
                        is_drum, # whether the track is a drum
                        track_name if len(track_name) > 0 else None, # name of track
                        has_lyrics, # whether the track has lyrics
                    )))],
                    columns = STEMS_TABLE_COLUMNS,
                ).to_csv( # append line to stems output filepath
                    path_or_buf = stems_output_filepath, sep = ",", na_rep = NA_STRING, header = False, index = False, mode = "a",
                )
                del program, is_drum, track_name, has_lyrics # free up memory

            # free up memory
            del midi

            ##################################################

            # SYNTHESIZE STEMS AS SAFETENSOR IF NECESSARY
            ##################################################

            # synthesize waveforms if necessary
            if need_to_synthesize:

                # initialize list of waveforms
                waveforms = [None] * len(track_paths)
                max_waveform_length = -1 # keep track of longest tensor (most samples)

                # loop through stems
                for j, track_path in enumerate(track_paths): 
                # for j, track_path in tqdm(iterable = enumerate(track_paths), desc = "Generating Waveforms for Stem", total = len(track_paths)):
                    waveform = get_waveform_tensor(path = track_path) # get waveform for this stem
                    if waveform.shape[-1] > MAX_N_SAMPLES_IN_STEM: # truncate overly long waveforms
                        waveform = waveform[:, :MAX_N_SAMPLES_IN_STEM]
                    if waveform.shape[-1] > max_waveform_length: # update waveform length tracker
                        max_waveform_length = waveform.shape[-1]
                    waveforms[j] = waveform # add waveform to stems tensors dictionary
                    remove(track_path) # remove temporary midi file
                    del waveform # free up memory
                
                # free up memory
                temp_dir.cleanup() # delete temporary directory properly
                del track_paths, temp_dir

                # transform waveforms
                for j in range(len(waveforms)):
                    waveforms[j] = torch.nn.functional.pad( # zero pad the end so all stems are the same length
                        input = waveforms[j], 
                        pad = (0, max_waveform_length - waveforms[j].shape[-1]), 
                        mode = "constant", 
                        value = 0,
                    )
                    waveforms[j] = waveforms[j].type(torch.float) / torch.max(input = waveforms[j]) # peak normalize

                # save safetensor to output path
                tensors = {str(j): waveform for j, waveform in enumerate(waveforms)} # keys need to be strings in safetensors
                del waveforms, max_waveform_length # free up memory
                save_file(tensors = tensors, filename = path_output) # save safetensor
                del tensors # free up memory

            ##################################################

            # SAVE SONG INFO
            ##################################################

            # write line of song info to file
            song_info = dataset.loc[i].to_dict() # get row of dataset as dictionary
            song_info["path"] = path_output # set path to the new output path
            del song_info["path_output"], song_info["mid"]
            pd.DataFrame( # create dataframe with one row
                data = [song_info],
                columns = SONGS_TABLE_COLUMNS,
            ).to_csv( # append line to output filepath
                path_or_buf = output_filepath, sep = ",", na_rep = NA_STRING, header = False, index = False, mode = "a",
            )
            del song_info

            # add output path to completed paths
            completed_paths.add(path_output)

            ##################################################

    ##################################################


    # USE MULTIPROCESSING TO CALL FUNCTIONS
    ##################################################

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(
            iterable = pool.imap(
                func = synthesize_song_at_index,
                iterable = dataset.index,
                chunksize = CHUNK_SIZE,
            ),
            desc = "Generating Stems",
            total = len(dataset),
        ))

    ##################################################
    
##################################################