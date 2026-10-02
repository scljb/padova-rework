"""
auth: L. J. Barratt
date: 02/09/26
"""

#%% PACKAGES

# * EXTERNAL PACKAGES
from configparser   import ConfigParser
from pathlib        import Path
from importlib      import import_module
from os             import get_terminal_size
try:
    twidth = get_terminal_size().columns
except OSError:
    twidth = 40  # fallback if terminal size not available

import sys
import time

# * INTERNAL PACKAGES
from lib.io.base_io     import BaseIO as IO
from lib.etc.registries import SOLVER_REGISTRY

#%% CLASS
class OVA:

    def __init__(self, args):

        # # CHECK FOR PARALLEL CLI ARGUMENT
        comm = None
        self.rank = 0
        size = 0

        if args.parallel is True:
            from mpi4py import MPI
            comm      = MPI.COMM_WORLD
            self.rank = comm.Get_rank()
            size      = comm.Get_size()

        # # FIND main.cfg IN GIVEN CASE DIRECTORY
        # ? get case directory path (relative to run.py)
        cwd = Path.cwd()
        case_directory = Path(f'{cwd}/cases/{args.case_name}')

        # ? check if main.cfg exists in case direcotry
        if not case_directory.exists():
            raise FileNotFoundError(
                    f'No main.cfg found in {case_directory}'
            )

        # ? read main.cfg
        config = ConfigParser(inline_comment_prefixes=('#', ';'))
        config.read(f'{case_directory}/main.cfg')
        
        # get solver info
        solution  = IO._parse_section(config, 'solution')   # parse section
        solver_id = solution['solver'][0]                   # solver name
        pars = ['time', 'write_trigger', 'write_increment']
        solver_args = {
            par: solution[par][0] for par in pars
        }

        # get parameters for primary inflow condition
        inflow = IO._parse_section(config, 'inflow')

        # cfg input files for models
        solver_inputs = solution['input']
        input_path    = {}
        for input in solver_inputs:

            model, file = input.split(':')
            input_path[input] = Path(f'{case_directory}/{file}.cfg')

            if not input_path[input].exists():
                raise FileNotFoundError(
                    f'No {input}.cfg found in {case_directory}'
                )
            

        # # CREATE INSTANCE OF THE IDENTIFIED SOLVER CLASS
        # ? get solver class information
        module_path, class_name = SOLVER_REGISTRY[solver_id]['class']

        # ? import module and get class attribute
        module = import_module(module_path)
        solver = getattr(module, class_name)

        # ? init solver class
        self.display_solver_text(class_name)
        sol = solver(
            input_path=input_path, 
            solver_args=solver_args,
            inflow_params=inflow,
            args=args,
            comm=comm
        )

        # TODO CLI class to deal with user-interaction and terminal display.
        # TODO maybe this could be in the OVA class.

        time_start = time.time()

        # # CALL SYSTEM SOLVER
        # ? call init method, if present
        if hasattr(sol, 'initialise_solver'):
            self.display_init_text()
            sol.initialise_solver()

        # ? evolve time until end time
        self.display_sim_text()
        sol.time_evolution(sol._define_progress_bar(args.progress, twidth))

        time_end = time.time()
        run_time = time_end - time_start
        sol.io._write_simulation_stats(class_name, run_time, size)

        # ? plot solutions
        # TODO create a class to handle post processing
        # TODO can pass sol to the plotter, use cli argument to choose

        # # END
        self.display_end_text()


    def display_solver_text(self, class_name):
        if self.rank == 0:
            print(
                '\n'+
                '#'*twidth+                          
                'PADOVA'.center(twidth)+   
                ('-'*(twidth//2)).center(twidth)+
                f'{class_name} Solver'.center(twidth)+
                '#'*twidth+  
                '\n'
            )
        

    def display_init_text(self):
        if self.rank == 0:
            print(
                '\n'+
                '#'*twidth+                          
                'Initialising solution.'.center(twidth)+   
                '#'*twidth+  
                '\n'
            )


    def display_sim_text(self):
        if self.rank == 0:
            print(
                '\n'+
                '#'*twidth+                          
                'Beginning simulation.'.center(twidth)+   
                '#'*twidth+  
                '\n'
            )


    def display_end_text(self):
        if self.rank == 0:
            print(
                '\n'+
                '#'*twidth+                          
                'End of program.'.center(twidth)+   
                '#'*twidth+  
                '\n'
            )








