"""
auth: L. J. Barratt
date: 28/09/26


"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
import numpy
from sys   import exit

# * INTERNAL LIBRARIES
from lib.solvers.base_solver     import BaseSolver
from lib.solvers.windkessel_tone import WindkesselTone
from lib.solvers.skeletal_muscle import SkeletalMuscle

# unit conversions
M3_PER_S_TO_L_PER_MIN = 60000

#%% CLASS

class WindkesselMuscle(BaseSolver):

    models = [
        'VariableWindkessel', 
        'VectorisedToneRegulation', 
        'OxygenTransport', 
        'CellularMetabolism'
    ]

    def configure_mpi(self, comm):

        super().configure_mpi(comm)

        if ( comm is not None ) and ( comm.Get_size() != 1 ):
            raise TypeError('SkeletalMuscle does not support MPI.')

    
    def create_system(self):

        # # WINDKESSEL + TONE
        self.wkt = WindkesselTone(
            comm=self.comm,
            io=self.io,
            tend=self.tend
        )


        # # OXYGEN TRANSPORT + METABOLISM
        self.mus = SkeletalMuscle(
            comm=self.comm,
            io=self.io,
            tend=self.tend
        )


        # # SOLUTION PARAMETERS
        self.t  = 0
        self.dt = min( [ self.wkt.dt, self.mus.dt ] ) #  time step size
        self.Nt = round( self.tend / self.dt )        # number of steps

        self.wkt_step_increment = round( self.wkt.dt / self.dt )
        self.mus_step_increment = round( self.mus.dt / self.dt )


    def initialise_solver(self, called=False):

        # # INITIALISE FLOW + TONE
        self.wkt.initialise_solver(called=True)
        muscle_flow = self.wkt.qout * M3_PER_S_TO_L_PER_MIN

        # # INITIALISE MUSCLE
        self.mus.O2.update_flow(muscle_flow)
        self.mus.update_atp_utilisation()
        self.mus.initialise_solver(called=True)

        # ? set basal tone using steady state oxygen
        self.wkt.tone.set_basal_state()

        # ? write initial condition
        if not called:
            self.m_attrs = self.get_model_dict(a=self)
            self.io.write_solutions(-1, 0, **self.m_attrs)


    def step(self, step):

        pass


    def update_tone_state(self):

        self.wkt.update_tone_state()

        # ? update oxygen contribution term
        P0    = 1
        alpha = 0.0013  # solubility coefficient of oxygen in blood
        pO2   = self.mus.O2.o2cap / alpha
        self.tone.met_term = P0 / ( P0 + pO2 )
