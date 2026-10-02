"""
auth: L. J. Barratt
Created on 11/10/25

Class housing common methods for input and output processing of 
PADOVA data.
"""

#%% PACKAGES

#* EXTERNAL PACKAGES
from os     import makedirs
from shutil import copytree, ignore_patterns
from configparser  import ConfigParser

#%% CLASS

class BaseIO():

    @staticmethod
    def configure_output(file):

        # # CREATE OUTPUT PATH
        case_path = file.parent
        output_dir = case_path / 'out_0/'
         
        counter = 1
        while output_dir.exists():
            output_dir = case_path / f'out_{counter}/'
            counter += 1


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
    def write_options(sol):
        write_trigger = sol['write_trigger'][0]

        if write_trigger == 'time':
            return round( 
                sol['write_increment'][0] / sol['dt'][0]
            )

        elif write_trigger == 'step':
            return sol['write_increment']

        else:
            raise ValueError('Write trigger options: time/step.')


    @staticmethod
    def parse_file(file, inputs):

        config = ConfigParser(inline_comment_prefixes=('#', ';'))
        config.read(file)

        return {
            section: BaseIO.parse_section(config, section)
            for section in inputs
        }

            
    @staticmethod
    def parse_section(config, section):
        """
        Parse a config section into a dict of floats or lists of 
        floats.
        """
        
        return {
            key: BaseIO.parse_value(value) 
            for key, value in config.items(section)
        }


    @staticmethod
    def parse_value(value):
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

