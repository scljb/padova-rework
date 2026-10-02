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

class OxygenIO(BaseIO):
        
    inputs  = ['solution', 'subject', 'blood']

    def write_solution(self, step, time, model, **stuff):
        
        cap = model.o2cap
        tis = model.o2tis

        with open(f'{self.output_dir}/times_o2t.txt', 'a') as res:
            if step == -1:
                res.write("NSTP \tTIME \to2capf \to2tisf\n")

            res.write(f'{step} \t{time:.6f} \t{cap:.8f} \t{tis:.8f}\n')