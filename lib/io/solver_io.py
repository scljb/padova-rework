"""
auth: L. J. Barratt
date: 14/09/26

Class housing methods for input and output processing.
"""

#%% PACKAGES

# * EXTERNAL PACKAGES
from importlib    import import_module
from os           import makedirs
from shutil       import copytree, ignore_patterns
from configparser import ConfigParser
import sys

#%% CLASS

class SolverIO:

    IO_REGISTRY = {
        'Artery':                   ('lib.io.artery_io',    'ArteryIO'),
        'MicrovascularNetwork':     ('lib.io.network_io',  'NetworkIO'),
        'ToneRegulation':           ('lib.io.tone_io',        'ToneIO'),
        'DirectToneRegulation':     ('lib.io.tone_io',        'ToneIO'),
        'RelativeToneRegulation':   ('lib.io.tone_io',        'ToneIO'),
        'VectorisedToneRegulation': ('lib.io.vec_tone_io', 'VecToneIO'),
        'Windkessel':               ('lib.io.windkessel_io', 'WindkesselIO'),
        'VariableWindkessel':       ('lib.io.var_wk_io',     'VarWKIO'),
        'OxygenTransport':          ('lib.io.oxygen_io',    'OxygenIO'),
        'CellularMetabolism':       ('lib.io.metabolism_io', 'MetabolismIO'),
    }

    def __init__(self, files, sargs, comm):

        self.comm = comm
        if comm is not None:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.rank = 0
            self.size = 1

        # # INSTANCE IO CLASSES
        self.model_io = {}
        self.inputs   = {}

        case_directory = None
        dts = []

        for key, path_to_file in files.items():

            model_name, file_name = key.split(':')

            # ? check if model has registered io class
            if model_name not in self.IO_REGISTRY:
                raise ValueError(
                    f'No IO class registered for model \'{model_name}\''
                )

            # ? instance io class for writes (done once per model type)
            # if model_name not in self.model_io:
            module_path, class_name = self.IO_REGISTRY[model_name]
            module = import_module(module_path)
            io_cls = getattr(module, class_name)
            self.model_io[model_name] = io_cls(path_to_file)

            # ? read file and store results (done for all input files)
            # section = self._parse_file(path_to_file)

            # ? store time steps
            model_dt = self.model_io[model_name].solution.get('dt', [None])[0]
            if model_dt is not None:
                dts.append( self.model_io[model_name].solution['dt'][0] )

            tmp = path_to_file.parent
            if case_directory is None:
                case_directory = tmp


        # # WRITE PARAMETERS
        # ? write increment (from minimum time step)
        self.write_increment = self.write_options(sargs, min(dts))

        # ? output directory
        self.output_dir = None
        if self.rank == 0:
            self.output_dir = self.configure_output(case_directory)

        # communicate direcotry to other ranks
        if self.comm is not None:
            self.output_dir = self.comm.bcast( 
                self.output_dir, 
                root=0 
            )

        # set in all io ub classes
        for _, cls in self.model_io.items():
            cls.output_dir = self.output_dir


    def write_solutions(self, step, time, **models):

        if ( not step % self.write_increment ) or ( step == -1 ):
            for model_name, io in self.model_io.items():

                model = models.get(model_name)

                if model is None:
                    continue

                io.write_solution(step, time, model, rank=self.rank)


    def configure_output(self, case_path):

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


    @staticmethod
    def write_options(sargs, dt):
        write_trigger = sargs['write_trigger']

        if write_trigger == 'time':
            return round( 
                sargs['write_increment'] / dt
            )

        elif write_trigger == 'step':
            return sargs['write_increment']

        else:
            raise ValueError('Write trigger options: time/step.')


    def _write_simulation_stats(self, solver, run_time, comm_size):
        with open(f'{self.output_dir}/runtime.txt', 'w') as f:
            f.write(
                f'{'Solver:':21} {solver}                  \n'
                f'{'Number of processes:':21} {comm_size}  \n'
                f'{'Total time:':21}{run_time/60:.4f} min  \n'
            )


    def _parse_file(self, file_path):
        config = ConfigParser(inline_comment_prefixes=('#', ';'))
        config.read(file_path)
        return {
            section: self._parse_section(config, section)
            for section in self.inputs
        }

        # for section in self.inputs:
        #     setattr(self, section, self._parse_section(config, section))


    def _parse_section(self, config, section):
        """
        Parse a config section into a dict of floats or lists of 
        floats.
        """
        
        return {
            key: self._parse_value(value) 
            for key, value in config.items(section)
        }


    @staticmethod
    def _parse_value(value):
        """
        Parse a config value into a list (of floats if possible).
        """
        
        value = value.strip()
        parts = [v.strip() for v in value.split(',')]
        
        # Try converting to float; if fails, return all as strings
        try:
            return [float(p) for p in parts]
        except ValueError:
            return parts  # fallback: list of strings
        
        
