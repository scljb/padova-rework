"""
auth: L. J. Barratt
date: 14/09/26

Class housing common methods for input and output processing of 
PADOVA data.
"""

#%% PACKAGES

#* EXTERNAL PACKAGES
from configparser  import ConfigParser

#%% CLASS

class BaseIO:

    def __init__(self, file):
        self.file       = file
        self.output_dir = None
        self._parse_file()


    def write_solution(self, *stuff):
        raise NotImplementedError


    def _parse_file(self):
        config = ConfigParser(inline_comment_prefixes=('#', ';'))
        config.read(self.file)
        for section in self.inputs:
            setattr(self, section, self._parse_section(config, section))

            
    @staticmethod
    def _parse_section(config, section):
        """
        Parse a config section into a dict of floats or lists of 
        floats.
        """
        
        return {
            key: BaseIO._parse_value(value) 
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