"""
auth: L. J. Barratt
Created on 27/03/26

Class housing MPI-related methods.
"""

#%% PACKAGES

#* EXTERNAL LIBRARIES
from mpi4py         import MPI
from configparser   import ConfigParser
from itertools      import combinations

#* INTERNAL LIBRARIES
from lib.io.base_io  import BaseIO


#%% CLASS

class BaseMPI:

    def __init__(self, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()


    def subcomm_lead_gid(self, subcomm):
        '''
        Get the global rank id of the lead rank (rank 0) of subcomm,
        and reduce it to all ranks via gcomm.
        
        Parameters
        ----------
        subcomm : MPI.Comm or None
            The sub-communicator whose rank-0 process is the leader.
            Ranks where subcomm is None are treated as non-leaders.

        Returns
        -------
        int
            Global rank id of the lead rank of subcomm.
        '''

        # get global id
        gid = self.comm.rank

        # identify subcomm owner status
        if ( subcomm is not None ) and ( subcomm.rank == 0 ):
            leader = gid

        else:
            leader = -1

        # send owner id to all ranks present
        leader_gid = self.comm.allreduce( leader, op=MPI.MAX )
        
        return leader_gid
    

    def communication_flags(self, **models):
        """Return communication flags for every pair of models."""

        # get global id
        gid = self.comm.rank

        # Determine the lead rank for each model
        leads = {
            name: self.subcomm_lead_gid(model.comm)
            for name, model in models.items()
        }

        flags = {}

        for a, b in combinations(models, 2):
            la = leads[a]
            lb = leads[b]

            flags[(a, b)] = (
                la != lb and
                gid in (la, lb)
            )

        return leads, flags


    @staticmethod
    def dummy_barrier(*args, **kwargs):
        return

















