"""
auth: L. J. Barratt
Created on 14/09/26

Class housing common methods for input and output processing of 
PADOVA data.
"""

#%% PACKAGES

#* EXTERNAL PACKAGES
from os     import makedirs
from shutil import copytree, ignore_patterns


#%% METHODS

def configure_output(case_path):

    # # GENERATE OUTPUT DIRECOTRY NAME
    output_dir = case_path / 'out_0/'
    counter = 1
    while output_dir.exists():
        output_dir = case_path / f'out_{counter}/'
        counter += 1

    # # MAKE DIRECTORY
    makedirs(output_dir, exist_ok=False)

    # # COPY CASE FILES TO OUTPUT DIRECTORY
    copytree(
        case_path, 
        output_dir / 'case/', 
        dirs_exist_ok=False,
        ignore=ignore_patterns('out_*')
    )

    return output_dir
