"""
auth: L. J. Barratt
date: 25/02/26

Runner script to call PADOVA solvers.
"""

#%% PACKAGES

#* EXTERNAL LIBRARIES
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ufl")
from argparse import ArgumentParser

#* INTERNAL LIBRARIES
from lib.ova import OVA as ova


#%% MAIN

if __name__ == "__main__":
    
    parser = ArgumentParser(
        description='PADOVA numerical solver.'
    )
    parser.add_argument(
        '-c', '--case_name', type=str, required=True,
        help='name of case directory which includes ova input files in .cfg format'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true', default=False,
        help='enable detailed terminal output'
    )
    parser.add_argument(
        '-p', '--progress', action='store_true', default=False,
        help='enable progress bar during simulation'
    )
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='enable parallelisation using mpi4py. reads from [MPI] section in settings.cfg in case directory'
    )
    args = parser.parse_args()

    # Call OVA startup class
    sim = ova(
        args
    )
