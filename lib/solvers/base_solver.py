"""
auth: L. J. Barratt
date: 14/09/26
"""

#%% PACKAGES

#* EXTERNAL LIBRARIES
from tqdm import tqdm
import sys

#* INTERNAL LIBRARIES
from lib.io.solver_io  import SolverIO 
from lib.etc.time_function import TimeFunction

#%% CLASS

class BaseSolver:

    required_attrs = ['t', 'dt', 'Nt']

    def __init__(
            self, 
            input_path=None, 
            solver_args=None, 
            inflow_params=None,
            args=None, 
            comm=None, 
            io=None,
            tend=None
    ):

        self.CLARGS = args       # command-line arguments
        self._tty_stream = None
        self.configure_mpi(comm)

        if io is None:

            # ? read input files
            self.tend = solver_args['time']
            self.read_input_file(input_path, solver_args)

            # ? setup root/inflow condition as time-dependent condition
            self.rc = TimeFunction(inflow_params)

        else:

            self.tend = tend
            self.io   = io

        self.create_system()
        self._check_required_attrs()

        # ? only needed on top-level solver
        # if io is None:
        #     self.m_attrs = self.get_model_dict(a=self)


    def create_system(self):
        raise NotImplementedError


    # def initialise_solver(self):
    #     raise NotImplementedError


    def step(self, **stuff):
        raise NotImplementedError
        

    def configure_mpi(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank() if comm is not None else 0


    def read_input_file(self, input_path, solver_args):
        
        for key, file in input_path.items():
            model_name, file_name = key.split(':')
            if model_name not in self.models:
                raise TypeError(
                    f'{model_name} is not used in {self.__class__} solver.'
                )


        self.io = SolverIO(input_path, solver_args, self.comm)

        if self.rank == 0:
            print(f'Output folder name: {self.io.output_dir}')


    def get_model_dict(self, **solvers):

        found = {}
        for solver in solvers.values():

            if solver is None:
                continue

            found.update(solver._get_models())


        return {name: found.get(name) for name in self.models}


    def time_evolution(self, pbar):

        for step in pbar:

            pbar.set_postfix(t=self._postfix())

            # # STEP SYSTEM IN TIME
            self.step(step)

            # # WRITE SOLUTION
            self.io.write_solutions(step, self.t, **self.m_attrs)


    def _postfix(self):
        return f'{self.t:.5f}s'


    def _get_models(self):
        '''
        Scan this solver's own attributes and return the ones whose
        class name matches an entry in self.models, as {class_name: instance}.

        This method requires the list of the model class names stored as a self.models. 
        
        Plain instance attributes must be present whose class __name__ matches an entry in self.models.
        '''

        found = {}
        for name in self.models:

            for attr_value in self.__dict__.values():

                if attr_value.__class__.__name__ == name:
                    found[name] = attr_value
                    break


        return found


    def _check_required_attrs(self):

        missing = [
            a for a in self.required_attrs
            if not hasattr(self, a) or getattr(self, a) is None
        ]

        if missing:
            raise ValueError(
                f'{type(self).__name__}: required attribute(s) not set: {missing}'
            )


    def _get_progress_stream(self):
        '''
        Return a stream that bypasses mpirun's stdout I/O forwarding,
        so tqdm's postfix updates show up live instead of buffering until exit.
        '''

        if self._tty_stream is None:
            try:
                self._tty_stream = open('/dev/tty', 'w')
            except OSError:
                # No controlling tty (e.g. batch job/scheduler) - fall back
                self._tty_stream = sys.stdout

        return self._tty_stream


    def _define_progress_bar(self, progress, twidth):
        return tqdm(
            range(self.Nt), 
            desc='Processing', 
            disable=not (progress and self.rank == 0),
            mininterval=0,
            file=self._get_progress_stream(),
            ncols=twidth
        )


    @staticmethod
    def extract_values(source_dict, keys, ii=0):
        '''
        Extract single values (indexed by ii) for the given keys from a dict
        of arrays.
        '''

        return {
            key: values[ii] 
            for key, values in source_dict.items() 
            if key in keys
        }
    

    @staticmethod
    def extract_all_values(source_dict, ii=0):
        '''
        Extract single values from a dict of arrays (indexed by ii).
        '''

        return {
            key: values[ii] 
            for key, values in source_dict.items()
        }
    