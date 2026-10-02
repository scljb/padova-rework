"""
auth: L. J. Barratt
Created on 08/05/25

Class housing methods for input and output processing.
"""

#%% PACKAGES

#* EXTERNAL PACKAGES
from numpy import column_stack, savetxt


#* INTERNAL PACKAGES
from lib.io.base_io import BaseIO


#%% CLASS

class AutoIO(BaseIO):

    def __init__(self, file:list):

        # # STORE VARIABLES
        self.file    = file

        # intended section names in config files
        self.net_inputs  = [
            'solution', 'fluid', 'inflow', 'outflow', 'network'
        ]
        self.ton_inputs  = [
            'solution', 'diameter', 'activation', 'contributions'
        ]

        # # READ FILE
        self.net = self.parse_file(self.file[0], self.net_inputs)
        self.ton = self.parse_file(self.file[1], self.ton_inputs)

        self.output_dir = self.configure_output(self.file[0])

        self.write_options(self.net['solution'])
        self.write_options(self.ton['solution'])


    def write_solution(self):
        


    def write_solution_avg_to_file(
            self, time, nlvls, pavg, vlvls, qavg, tavg, qtot
    ):

        # formatting
        ndata = column_stack((nlvls, pavg))
        vdata = column_stack((vlvls, qavg, qtot, tavg))

        # print to file
        filename = f'./out/{self.output_dir}/level_averaged_data/'
        savetxt(f'{filename}node_avgs_t{time:.6f}.txt', ndata, header='nlvl pavg', comments='')
        savetxt(f'{filename}vessel_avgs_t{time:.6f}.txt', vdata, header='vlvl qavg qtot tavg', comments='')


    def write_tree_properties_to_file(self, nstp, time, rtot, ctot):
        with open(f'./out/{self.output_dir}/properties.txt', 'a') as res:
            if nstp == 0:
                res.write("NSTP\t\tTIME\t\trtot\t\tctot\n")

            res.write(f'{nstp}\t\t{time:.4f}\t\t{rtot:.6e}\t\t{ctot:.6e}\n')


    