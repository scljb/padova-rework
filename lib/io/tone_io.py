"""
auth: L. J. Barratt
Created on 15/09/26

Class housing methods for input and output processing.
"""

#%% PACKAGES

#* INTERNAL PACKAGES
from lib.io.base_io import BaseIO


#%% CLASS

class ToneIO(BaseIO):
        
    inputs  = ['solution', 'diameter', 'activation', 'contributions']

    def write_solution(self, step, time, model, **stuff):

        lmbda = model.lmbda
        theta = getattr(model, 'theta', -1)

        with open(f'{self.output_dir}/tone.txt', 'a') as res:
            if step == -1:
                res.write("NSTP \tTIME \tlmbda \tlmbda\n")

            res.write(f'{step} \t{time:.6f} \t{lmbda:.6f}\t{theta:.6f}\n')