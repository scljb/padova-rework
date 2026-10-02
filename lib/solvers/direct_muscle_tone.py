"""
auth: L. J. Barratt
date: 15/09/26


"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
import sys
from mpi4py.MPI import UNDEFINED


# * INTERNAL LIBRARIES
from lib.solvers.direct_network_tone import DirectNetworkTone
from lib.solvers.skeletal_muscle     import SkeletalMuscle

from lib.solvers.base_solver import BaseSolver

#%% CLASS

class DirectMuscleTone(BaseSolver):

    models = [
        'MicrovascularNetwork', 
        'DirectToneRegulation', 
        'OxygenTransport', 
        'CellularMetabolism'
    ]

    def configure_mpi(self, comm):

        super().configure_mpi(comm)

        # ? create a subcomm for the muscle model
        if self.comm is not None:
            color = 0 if self.rank == 0 else UNDEFINED
            self.mcomm = self.comm.Split(color=color, key=0)
        else:
            self.mcomm=None
        


    def create_system(self):

        # # CONFIGURE TONE-REGULATING NETWORK
        self.bed = DirectNetworkTone(
            comm=self.comm,
            io=self.io,
            tend=self.tend
        )

        # # CONFIGURE OXYGEN TRANSPORT & METABOLISM
        if self.rank == 0:
            self.mus = SkeletalMuscle(
                comm=self.mcomm,
                io=self.io,
                tend=self.tend
            )   

        else:
            self.mus = None


        # # SHARE DATA ON ALL RANKS
        bed_dt = self.bed.dt
        if ( self.comm is not None ) and ( self.comm.Get_size() > 1 ):
            mus_dt = self.comm.bcast( 
                self.mus.dt if self.rank == 0 else None, 
                root=0 
            )
        else:
            mus_dt = self.mus.dt


        # # SOLUTION PARAMETERS
        self.t  = 0
        self.dt = min( [bed_dt, mus_dt] )       #  time step size
        self.Nt = round( self.tend / self.dt )  # number of steps

        self.bed_step_increment = round( bed_dt / self.dt )
        self.mus_step_increment = round( mus_dt / self.dt )

        # # COUPLED PARAMETERS
        self._flow_curr    = 0   # network to muscle
        self._oxygen_curr  = 0   # muscle to tone
        self._volume_curr  = 0   # blood volume in network


    def initialise_solver(self):

        # ? solve network model in initial state
        self.bed.update_root_condition(force=True)
        self.tension_curr, self.shear_curr, self.flow_curr = \
                self.compute_network_state()

        # ? solve muscle solver until steady state
        if self.rank == 0:
            self.mus.O2.update_flow(self.flow_curr*60000)
            self.mus.initialise_solver(called=True)
            self.oxygen_curr = self.mus.O2.o2cap


        # ? update reference values in tone model
        if self.rank == 0:
            self.bed.tone.update_reference_tension(self.tension_curr)
            self.bed.tone.update_reference_shear(self.shear_curr)
            self.bed.tone.update_reference_oxygen(self.oxygen_curr)


        # ? write initial condition
        self.m_attrs = self.get_model_dict(bed=self.bed, mus=self.mus)
        self.io.write_solutions(-1, 0, **self.m_attrs)


    def step(self, step):

        # # SOLVE TONE-REGULATING NETWORK
        # * Micro-vessel Model
        # ? update tone-dependent conditions
        self.bed.update_network_tone()

        # ? update root boundary condition
        self.bed.update_root_condition(force=self.bed._network_tone)

        # ? solve
        self.tension_curr, self.shear_curr, self.flow_curr = \
                self.compute_network_state()

        # * Tone Model
        if ( self.bed.tone is not None ) and \
                ( not step % self.bed_step_increment ):        

            # ? update current values for tone
            self.bed.tone.update_current_tension(self.tension_curr)
            self.bed.tone.update_current_shear(self.shear_curr)
            self.bed.tone.update_current_oxygen(self.oxygen_curr)

            # ? solve
            self.bed.tone.solve()


        # ? update local time
        self.bed.t += self.bed.dt


        # # SOLVE SKELETAL MUSCLE
        if ( self.mus is not None ) and ( not step % self.mus_step_increment ):

            # ? update time-dependent parameters
            self.mus.O2.update_flow(self.flow_curr*60000)
            self.mus.O2.update_blood_volume(self.vol_curr)
            self.mus.update_atp_utilisation()

            # ? solve
            self.mus.O2.solve()
            self.mus.CM.solve()

            # ? compute rate of oxphos
            self.mus.update_oxygen_utilisation()

            # ? free capillary oxygenation
            self.oxygen_curr = self.mus.O2.o2cap

            # ? update local time
            self.mus.t += self.mus.dt


        # # UPDATE TIME
        self.t += self.dt


    def compute_network_state(self):

        tension, shear = self.bed.compute_network_state()
        flow = self.bed.network.compute_root_flow()

        # todo compute blood volume, feed to muscle 
        volume = self.bed.network.compute_tree_volume()
            # todo adapt this function to compute CAPILLARY volume

        return tension, shear, flow


    @property
    def tension_curr(self):
        return self.bed._tension_curr

    @tension_curr.setter
    def tension_curr(self, value):
        self.bed._tension_curr = value

    @property
    def shear_curr(self):
        return self.bed._shear_curr

    @shear_curr.setter
    def shear_curr(self, value):
        self.bed._shear_curr = value

    @property
    def lmbda_curr(self):
        return self.bed._lmbda_curr

    @lmbda_curr.setter
    def lmbda_curr(self, value):
        self.bed._lmbda_curr = value

    @property
    def oxygen_curr(self):
        return self._oxygen_curr

    @oxygen_curr.setter
    def oxygen_curr(self, value):
        self._oxygen_curr = value

    @property
    def flow_curr(self):
        return self._flow_curr

    @flow_curr.setter
    def flow_curr(self, value):
        self._flow_curr = value

    @property
    def volume_curr(self):
        return self._volume_curr

    @volume_curr.setter
    def volume_curr(self, value):
        self._volume_curr = value






        