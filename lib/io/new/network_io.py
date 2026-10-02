"""
auth: L. J. Barratt
Created on 14/09/26

Class housing methods for input and output processing.
"""

#%% PACKAGES

#* EXTERNAL PACKAGES
from numpy import column_stack, savetxt

#* INTERNAL PACKAGES
from lib.io.new.base_io import BaseIO


#%% CLASS

class NetworkIO(BaseIO):
        
    inputs  = ['solution', 'fluid', 'outlet', 'network']

    def write_solution(self, step, time, model, rank):

        nlvls_arr, vlvls_arr, pavg, qavg, tavg, qtot = \
                model.compute_level_averages()

        if rank == 0:
            self.write_solution_average(
                    time, 
                    nlvls_arr, vlvls_arr, 
                    pavg, 
                    qavg, tavg, qtot
            )


    def write_solution_average(
            self, time, nlvls, vlvls, pavg, qavg, tavg, qtot
    ):

        # formatting        
        ndata = column_stack((nlvls, pavg))
        vdata = column_stack((vlvls, qavg, qtot, tavg))

        # print to file
        filename = f'{self.output_dir}/level_averaged_data/'
        savetxt(f'{filename}node_avgs_t{time:.6f}.txt', ndata, header='nlvl pavg', comments='')
        savetxt(f'{filename}vessel_avgs_t{time:.6f}.txt', vdata, header='vlvl qavg qtot tavg', comments='')


    def write_tree_properties(self, nstp, time, rtot, ctot):
        with open(f'{self.output_dir}/peripheral_properties.txt', 'a') as res:
            if nstp == -1:
                res.write("NSTP\t\tTIME\t\trtot\t\tctot\n")

            res.write(f'{nstp}\t\t{time:.4f}\t\t{rtot:.6e}\t\t{ctot:.6e}\n')

