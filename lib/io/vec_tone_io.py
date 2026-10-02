"""
auth: L. J. Barratt
date: 30/09/26
"""

#%% PACKAGES
import numpy as np

#* INTERNAL PACKAGES
from lib.io.base_io import BaseIO


#%% CLASS

class VecToneIO(BaseIO):
        
    inputs  = ['solution', 'time_constants']

    def write_solution(self, step, time, model, **stuff):
        lmbda = np.asarray(model.lmbda)
        theta = np.asarray(getattr(model, 'theta', -1))

        # # WRITE LMBDA SOLUTIONS
        with open(f'{self.output_dir}/lmbda.txt', 'a') as res:

            if step == -1:
                header = "NSTP\tTIME"
                header += ''.join(
                    f'\tlmbda_g{g+1}' for g in range(len(lmbda))
                )
                res.write(header + '\n')

            row = f'{step}\t{time:.6f}'
            row += ''.join(f'\t{x:.6f}' for x in lmbda)
            res.write(row + '\n')

        # # WRITE THETA SOLUTIONS
        with open(f'{self.output_dir}/theta.txt', 'a') as res:

            if step == -1:
                header = "NSTP\tTIME"
                header += ''.join(
                    f'\ttheta_g{g+1}' for g in range(len(theta))
                )
                res.write(header + '\n')

            row = f'{step}\t{time:.6f}'
            row += ''.join(f'\t{x:.6f}' for x in theta)
            res.write(row + '\n')
    

    def old_write_solution(self, step, time, model, **stuff):

        lmbda = model.lmbda
        theta = model.theta

        with open(f'{self.output_dir}/tone.txt', 'a') as res:
            if step == -1:
                res.write("NSTP \tTIME \tlmbda \ttheta \n")

            res.write(f'{step} \t{time:.6f} \t{lmbda:.6f}\t{theta:.6f}\n')