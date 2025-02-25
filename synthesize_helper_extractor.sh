#!/bin/bash

# README
# Phillip Long
# February 16, 2025

# Extract tarballs in sPDMX stems.

# sh /home/pnlong/synthesize_audio/synthesize_helper_extractor.sh /deepfreeze/user_shares/pnlong/sPDMX_stems

# SETUP
##################################################

# stop execution if there is an error
set -e

# get directories
base_directory=${1}
data_directory="${base_directory}/data"

# get parent directories, sort so it is cleaner
subdirectories=($(ls ${data_directory} | sed "s+^+${data_directory}/+"))
IFS=$'\n' # https://stackoverflow.com/questions/7442417/how-to-sort-an-array-in-bash
subdirectories=($(sort <<<${subdirectories[*]})) # sort parent directories so it is cleaner
unset IFS # unset IFS

# separator line function
function separator_line {
    printf "=%.0s" {1..100}
    printf "\n"
}

# output important information
n_subdirectories=(${#subdirectories[@]})
printf "${n_subdirectories} subdirectories.\n"

##################################################


# ITERATE THROUGH SUBDIRECTORIES
##################################################

# iterate through parent directories
for i in "${!subdirectories[@]}"; do

    # add separator line
    separator_line
    
    # get directory
    subdirectory=${subdirectories[i]}
    cd ${subdirectory} # change directory to parent directory

    # get list of child tar files
    tarballs=($(find . -name "*.tar"))

    # output information about number of tarballs in subdirectory
    subdirectory_name=($(basename ${subdirectory}))
    n_tarballs=(${#tarballs[@]})
    nth_subdirectory=$((i + 1))
    printf "${subdirectory_name} (${nth_subdirectory}/${n_subdirectories}): ${n_tarballs} tarballs to extract.\n"

    # skip further calculations if there are no tarballs in a directory
    if [ ${n_tarballs} -eq 0 ]; then
        continue
    fi

    # iterate through tarballs
    for tarball in ${tarballs[@]}; do

        # get variables
        tarball=($(basename ${tarball})) # get just basename of tarball

        # extract tarball
        tarball_extracted_dir=(${tarball%%.*}) # relative path to resulting extracted directory, removing file extension
        if [ ! -d ${tarball_extracted_dir} ]; then # check that resulting extracted directory doesn't exist already (avoid redundant extractions)
            if [[ ${tarball} == *.tar ]]; then # for normal tarballs
                # tar -xf "${tarball}"
                sleep .2
            else # for gzipped tarballs
                # tar -xfz "${tarball}"
                sleep .2
            fi
        fi

        # print a period when this tarball is successfully extracted
        printf "." # print . when tar is extracted

    done

    # print final newline
    printf "\n"

done

# print separator line at the end
separator_line

##################################################
