"""
auth: L. J. Barratt
date: 30/09/26
"""

#%% PACKAGES
from lib.io.base_io import BaseIO


#%% CLASS

class VarWKIO(BaseIO):
        
    inputs  = ['solution', 'fluid', 'fractal']

    def write_solution(self, step, time, model, **stuff):

        pcap = model.pcap
        qin  = ( model.pin - pcap )  / model.Rn
        qout = ( pcap - model.pout ) / model.RT
        id   = model.windkessel_id

        # ? solution variables
        with open(f'{self.output_dir}/wk_{id}.txt', 'a') as res:
            if step == -1:
                res.write("NSTP \tTIME \tpcap \tqin \tqout\n")

            res.write(
                f'{step} \t{time:.6f} \t{pcap:.8f} \t{qin:.8e} \t{qout:.8e}\n'
            )

        # ? tree properties
        with open(f'{self.output_dir}/wk_{id}_prop.txt', 'a') as res:
            if step == -1:
                res.write("NSTP \tTIME \tRn \tRm \tRf \tC\n")

            res.write(
                f'{step} \t{time:.6f} \t{model.Rn:.8e} \t{model.Rm:.8e} \t{model.Rf:.8e} \t{model.C:.8e}\n'
            )