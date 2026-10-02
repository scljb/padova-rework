"""
auth: L. J. Barratt
Created on 08/05/25

Class housing methods for input and output processing.
"""

#%% PACKAGES

#* INTERNAL PACKAGES
from lib.io.artery_io import ArteryIO


#%% CLASS

class ArteryWindkesselIO(ArteryIO):
    
    def __init__(self, file):
        super().__init__(file)


    def write_wk_solution_to_file(self, step, wk, idx, time):

        # # PROCESS DATA
        qin  = wk.qin
        qout = wk.qout
        pc   = wk.pcold

        # # PRINT TO FILE
        filename = self.output_dir / f'windkessel_for_artery_{idx}.txt'
        with open(filename, 'a') as res:
            if step == 0: 
                res.write("step\ttime\tqin\tqout\tpc\n")

            res.write(f'{step}\t{time:.6f}\t{qin:.6f}\t{qout:.6f}\t{pc:.6f}\n')