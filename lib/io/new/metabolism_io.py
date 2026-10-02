"""
auth: L. J. Barratt
Created on 14/09/26

Class housing methods for input and output processing.
"""

#%% PACKAGES

#* EXTERNAL PACKAGES

#* INTERNAL PACKAGES
from lib.io.new.base_io import BaseIO


#%% CLASS

class MetabolismIO(BaseIO):
        
    inputs  = ['solution', 'atpase', 'metabolism']

    def write_solution(self, step, time, model, **stuff):
        
        atp = model.atp
        pcr = model.pcr

        with open(f'{self.output_dir}/times_met.txt', 'a') as res:
            if step == -1:
                res.write("NSTP \tTIME \tATP \tPCr\n")

            res.write(f'{step} \t{time:.6f} \t{atp:.8f} \t{pcr:.8f}\n')