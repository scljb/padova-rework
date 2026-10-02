"""
auth: L. J. Barratt
Created on 27/03/26

Class housing MPI-related methods for the Macro model.
"""

#%% PACKAGES

from numpy import array

#* INTERNAL PACKAGES
from lib.mpi.base_mpi import BaseMPI


#%% CLASS

class ArteryMPI(BaseMPI):   

    def distribute_arteries(self, nart):
        """
        Assign arteries to ranks.

        Parameters
        ----------
        nart : int
            Total number of arteries.

        Returns
        -------
        local_arteries : list
            List of artery indices owned by this rank.
        ownership_dict : dict
            Dictionary {rank: [artery_ids]}
        """

        if self.size > nart:
            raise ValueError(
                f'In ArterialNetwork: the number of ranks '
                f'({self.size}) <= the number of arteries ({nart}).'
            )

        arts_per_rank = nart // self.size
        remainder = nart % self.size

        start = 0
        local_arteries = []
        ownership_dict = {}

        for r in range(self.size):
            n = arts_per_rank + (1 if r < remainder else 0)
            end = start + n
            arteries_r = list(range(start, end))
            ownership_dict[r] = arteries_r 

            if r == self.rank:
                local_arteries = arteries_r


            start = end


        return local_arteries, ownership_dict
    
