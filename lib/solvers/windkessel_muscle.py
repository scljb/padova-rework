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

        self.step_wkt = round( self.wkt.dt / self.dt )
        self.step_mus = round( self.mus.dt / self.dt )


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

        # ? write initial solution
        if not called:
            self.m_attrs = self.get_model_dict(a=self)
            self.io.write_solutions(-1, 0, **self.m_attrs)


    def step(self, step):

        # # SOLVE SYSTEMS
        # ? windkessel + tone model
        if not step % self.step_wkt:

            if not step % self.wkt.step_wk:
                self.update_windkessel_inflow()
                self.wk.apply_dilation(self.tone.lmbda)
                self.wk.solve()

            if not step % self.wkt.step_tone:
                self.update_tone_state()
                self.tone.solve()


            self.wkt.t += self.wkt.dt
            

        # ? oxygen + metabolism model
        if not step % self.step_mus:
            self.mus.O2.solve()
            self.mus.update_oxygen_utilisation()
            self.mus.CM.solve()
            self.mus.t += self.mus.dt


        # # UPDATE TIME
        self.t += self.dt
        

    def update_windkessel_inflow(self):
        new_root = self.rc.compute_value(self.t)
        self.wkt.wk.update_root_pressure( new_root )


    def update_tone_state(self):

        self.wkt.update_tone_state()

        # ? update oxygen contribution term
        P0    = 1
        alpha = 0.0013  # solubility coefficient of oxygen in blood
        pO2   = self.mus.O2.o2cap / alpha
        self.tone.met_term = P0 / ( P0 + pO2 )
