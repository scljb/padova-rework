"""
auth: L. J. Barratt
date: 22/09/26

Class housing methods for input and output processing.
"""

#%% PACKAGES
from lib.io.base_io import BaseIO


#%% CLASS

class WindkesselIO(BaseIO):
        
    inputs  = ['solution', 'vascular', 'fluid']

    def write_solution(self, step, time, model, **stuff):

        pcap = model.pcap
        qin  = ( model.pin - pcap )  / model.R1
        qout = ( pcap - model.pout ) / model.R2
        id   = model.windkessel_id

        # ? solution variables
        with open(f'{self.output_dir}/wk_{id}.txt', 'a') as res:
            if step == -1:
                res.write("NSTP \tTIME \tpcap \tqin \tqout\n")

            res.write(
                f'{step} \t{time:.6f} \t{pcap:.8f} \t{qin:.8f} \t{qout:.8f}\n'
            )

        # ? tree properties
        with open(f'{self.output_dir}/wk_{id}_prop.txt', 'a') as res:
            if step == -1:
                res.write("NSTP \tTIME \tR1 \tR2 \tC\n")

            res.write(
                f'{step} \t{time:.6f} \t{model.R1:.8e} \t{model.R2:.8e} \t{model.C:.8e}\n'
            )