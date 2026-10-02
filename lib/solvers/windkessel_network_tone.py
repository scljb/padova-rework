"""
auth: L. J. Barratt
date: 22/09/26


"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
from numpy  import exp
from mpi4py.MPI import UNDEFINED

# * INTERNAL LIBRARIES
from lib.solvers.base_solver           import BaseSolver
from lib.solvers.direct_network_tone   import DirectNetworkTone
from lib.models.windkessel             import Windkessel



#%% CLASS

class WindkesselNetworkTone(BaseSolver):

    models = ['MicrovascularNetwork', 'DirectToneRegulation', 'Windkessel']

    def configure_mpi(self, comm):

        super().configure_mpi(comm)

        # ? create a subcomm for the tone model
        if self.comm is not None:
            color = 0 if self.rank == 0 else UNDEFINED
            self.wkcomm = self.comm.Split(color=color, key=0)        
        else:
            self.wkcomm=comm


    def create_system(self):

        wk_io = self.io.models['Windkessel']

        # # CONFIGURE TONE-REGULATING NETWORK SOLVER
        self.bed = DirectNetworkTone(
            comm=self.comm,
            io=self.io,
            tend=self.tend
        )

        # # CONFIGURE WINDKESSEL MODEL
        self.wk = Windkessel(
            self.dt,
            self.y1,
            wk_io.vascular,
            wk_io.fluid
        )

        # # STORE PARAMETERS
        # * Time parameters
        self.t  = 0                                 # initial time [s]

        # ? require equal time steps
        if wk_io.solution['dt'][0] == self.bed.dt:
            self.dt   = wk_io.solution['dt'][0]     # Time step [sec]
            self.Nt = round( self.tend / self.dt )  # Number of steps
        else:
            raise ValueError('time steps do not match')
        
        # * Inflow condition
        if wk_io.inflow['method'][0] == 'exponential':
            self.y1         = wk_io.inflow['value'][0]  # root pressure 1 [Pa]
            self.y2         = wk_io.inflow['value2'][0] # root pressure 2 [Pa]
            self.delta      = wk_io.inflow['delta'][0]  # time delay [s]
            self.tau        = wk_io.inflow['tau'][0]    # time constant [s] 

        # * 


    def initialise_solver(self):
        '''
        Steady state given flow in and ATP demand.

        0. set root inflow
        0. set vascular properties
        loop:
        1. get muscle flow
        2. get shear + tension
        3. set reference shear, tension

        '''

        # ? find steady state for wk
        self.update_wk_inflow()
        self.wk.steady_state_solve()

        # ? write initial condition
        self.m_attrs = self.get_model_dict(a=self)
        self.io.write_solutions(-1, 0, **self.m_attrs)


    def time_evolution(self, pbar):

        for step in pbar:

            pbar.set_postfix(t=f'{self.t:.5f}s')

            # # SOLVE SYSTEMS
            self.step(step)

            # # WRITE SOLUTION
            self.io.write_solutions(step, self.t, **self.m_attrs)


    def step(self, step):

        # # SOLVE WINDKESSEL
        # ? update tone-dependent state
        self.update_wk_tone()

        # ? update time-dependent inflow
        self.update_wk_inflow()

        # ? solve
        self.wk.solve()

        # # SOLVE TONE REGULATION
        # ? update current values for tone
        # self.tone.update_current_tension(self.tension_curr)
        # self.tone.update_current_shear(self.shear_curr)

        # ? solve
        # self.tone.solve(self.t)

        # # UPDATE TIME
        self.t += self.dt


    def compute_network_state(self):

        tension, shear = self.bed.compute_network_state()

        rtot = self.bed.network.compute_tree_resistance()
        r1   = self.r1rt * rtot
        r2   = rtot - r1
        ceff = self.bed.network.compute_tree_compliance()
        

        return tension, shear, r1, r2, ceff


    def update_wk_inflow(self):

        # # COMPUTE NEW ROOT PRESSURE
        alpha = self.t >= self.delta
        new_root = (
            self.y1 + alpha * ( self.y2-self.y1 ) * ( 
                1.0 - exp( -(self.t - self.delta)/self.tau ) 
            )
        )

        # ? update root condition
        self.wk.update_root_pressure( new_root )


    def update_wk_tone(self):

        # # COMPUTE CHANGE IN PARAMETERS
        r1 = self.wk.R1_ref
        r2 = self.wk.R2_ref * self.tone.lmbda**(-4)
        c  = self.wk.C_ref  * self.tone.lmbda


        # # UPDATE PARAMETERS IN OBJECT
        self.wk.update_vascular_properties(r1, r2, c)