# README
# Phillip Long
# January 14, 2025

# Help synthesize data as audio.

# python /home/pnlong/synthesize_audio/synthesize_helper.py

# IMPORTS
##################################################

from safetensors.torch import save_file

import argparse
from os.path import exists, dirname, basename
from os import remove, mkdir, chdir
from shutil import rmtree
import subprocess
import pandas as pd
import torch
import numpy as np
import tempfile
import subprocess
from tqdm import tqdm
import logging
import pickle
import subprocess

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from synthesize_manager import NA_STRING, STEMS_TABLE_COLUMNS, SONGS_TABLE_COLUMNS, SAMPLE_RATE, GAIN, LINE
from model_musescore import load, MusicRender

##################################################


# ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Synthesize Helper", description = "Helper for synthesizing PDMX as audio waveforms stored in tensors.")
    parser.add_argument("-i", "--instructions_filepath", type = str, required = True, help = "Filepath to pickled instructions information")
    return parser.parse_args(args = args, namespace = namespace)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # parse the command-line arguments
    args = parse_args()

    # load in instructions
    with open(args.instructions_filepath, "rb") as instructions_file:
        instructions = pickle.load(instructions_file)
        dataset_filepath = instructions["dataset_filepath"]
        soundfont_filepath = instructions["soundfont_filepath"]
        output_filepath = instructions["output_filepath"]
        stems_output_filepath = instructions["stems_output_filepath"]
        subdirectories = instructions["subdirectories"]
        reset = instructions["reset"]
        reset_tables = instructions["reset_tables"]
        temporary_storage_dir = instructions["temporary_storage_dir"]
        use_tarball_buffer = instructions["use_tarball_buffer"]
        del instructions

    # set up logger
    logging.basicConfig(level = logging.INFO, format = "%(message)s")

    # load in dataset
    dataset = pd.read_csv(filepath_or_buffer = dataset_filepath, sep = ",", header = 0, index_col = False)
    
    # group paths into their subdirectories
    subdirectories_set = set(subdirectories) # for faster find operations
    subdirectories_to_indicies = dict() # for each subdirectory, store a list of the indicies of output paths for that subdirectory
    for i in dataset.index:
        subdirectory = dirname(dataset.at[i, "path_output"])
        if (subdirectory not in subdirectories_set):
            continue
        else:
            if subdirectory in subdirectories_to_indicies.keys():
                subdirectories_to_indicies[subdirectory].append(i)
            else:
                subdirectories_to_indicies[subdirectory] = [i]
    
    # free up memory
    del subdirectories_set # free up memory
    dataset = dataset.iloc[sum(subdirectories_to_indicies.values(), [])] # filter dataset down to necessary rows (only the relevant subdirectories)

    # get already written filepaths
    if exists(output_filepath):
        already_completed_paths = set(pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = None, names = SONGS_TABLE_COLUMNS, usecols = ["path"], index_col = False)["path"])
    else:
        already_completed_paths = set()

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
                "-i", soundfont_filepath,
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
                        tracks = [music.tracks[i]],
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
    def synthesize_song(i: int, path_output: str):
        """
        Given the dataset index and output path, read the song as a music object, and 
        generate a safetensor that contains the audio waveforms for each stem.
        Writes to files.
        """

        # save safetensor if necessary
        loaded_music = False # track whether the music object has been loaded
        create_safetensor = not exists(path_output) or reset
        if create_safetensor:

            # load music object
            music = load(path = dataset.at[i, "path"])
            loaded_music = True

            # save waveforms as file
            write_safetensor(music = music, path_output = path_output)

        # write tables info to file
        path_output = dataset.at[i, "path_output"] # ensure path output is correct, regardless of where the safetensor was written
        if path_output not in already_completed_paths:

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

            # add to already completed paths
            already_completed_paths.add(path_output)
            

    def synthesize_song_helper(i: int, path_output: str):
        """
        Helper function for synthesizing songs.
        """

        # try to synthesize song
        try:
            synthesize_song(i = i, path_output = path_output)
        
        # print exception if failure and exit program
        except Exception as exception:
            logging.info(exception)
            logging.info(f"The exception occurred at i = {j}")
            sys.exit(1)

    ##################################################


    # ITERATE OVER SUBDIRECTORIES
    ##################################################

    # using tarball buffer
    if use_tarball_buffer:

        # change directory to temporary storage directory
        chdir(temporary_storage_dir)

        # iterate over subdirectories
        for i, subdirectory in enumerate(subdirectories):

            # print a line
            logging.info(LINE)
            
            # get relevant indicies for subdirectory
            indicies_for_subdirectory = subdirectories_to_indicies[subdirectory]

            # name of temporary storage subdirectory
            temporary_storage_subdir_name = "-".join(subdirectory.split("/")[-2:]) # name of temporary directory

            # subdirectory tarball filepath (filepath to tarball on deepfreeze)
            subdirectory_tarball_filepath = f"{subdirectory}.tar"

            # message for this subdirectory
            message = f"{temporary_storage_subdir_name} ({i + 1} of {len(subdirectories)})"

            # check if this subdirectory is already complete
            if exists(subdirectory_tarball_filepath) and not (reset or reset_tables):
                logging.info(f"{message} already complete.")
                continue

            # get subdirectory temporary storage directory name
            temporary_storage_subdir = f"{temporary_storage_dir}/{temporary_storage_subdir_name}" # full path of temporary directory
            # if exists(temporary_storage_subdir):
            #     rmtree(temporary_storage_subdir, ignore_errors = True)
            # mkdir(temporary_storage_subdir)
            if not exists(temporary_storage_subdir):
                mkdir(temporary_storage_subdir)

            # iterate over indicies in subdirectory
            for j in tqdm(
                iterable = indicies_for_subdirectory,
                desc = message,
                total = len(indicies_for_subdirectory),
            ):
                synthesize_song_helper(i = j, path_output = f"{temporary_storage_subdir}/{basename(dataset.at[j, 'path_output'])}")

            # tar directory (no need to gzip, it's unclear how much compression will help)
            logging.info("Tarballing.")
            # chdir(temporary_storage_dir) # change working directory to temporary storage directory
            temporary_storage_subdir_tarball_filepath = f"{temporary_storage_subdir}.tar"
            if exists(temporary_storage_subdir_tarball_filepath):
                remove(temporary_storage_subdir_tarball_filepath)
            subprocess.run(args = ["tar", "-cf", basename(temporary_storage_subdir_tarball_filepath), basename(temporary_storage_subdir)], check = True)
            rmtree(temporary_storage_subdir) # remove temporary tree directory to save storage

            # move onto deepfreeze
            logging.info("Copying tarball to DeepFreeze.")
            subprocess.run(args = ["rsync", temporary_storage_subdir_tarball_filepath, subdirectory_tarball_filepath], check = True)
            # copy2(src = temporary_storage_subdir_tarball_filepath, dst = subdirectory_tarball_filepath)
            remove(temporary_storage_subdir_tarball_filepath) # remove tar file

            # untar file on deepfreeze
            # logging.info("Extracting tarball.")
            # chdir(dirname(subdirectory)) # change directory to on the nas
            # subprocess.run(args = ["tar", "-xf", basename(subdirectory_tarball_filepath)], check = True)
            # remove(subdirectory_tarball_filepath) # remove tar file on deepfreeze

            # change working directory back to temporary storage directory
            # chdir(temporary_storage_dir)
                
        # print a line
        logging.info(LINE)

    # writing directly to deepfreeze
    else:

        # concatenate indicies of all subdirectories together
        indicies_for_all_subdirectories = sum([subdirectories_to_indicies[subdirectory] for subdirectory in subdirectories], [])
        
        # iterate over all indicies in all given subdirectories
        for j in tqdm(
            iterable = indicies_for_all_subdirectories,
            desc = "Generating Stems",
            total = len(indicies_for_all_subdirectories),
        ):
            synthesize_song_helper(i = j, path_output = dataset.at[j, "path_output"])

    ##################################################
    
##################################################