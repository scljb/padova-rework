"""
auth: L. J. Barratt
date: 02/09/26


"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
from numpy  import isclose
from os     import makedirs
from mpi4py.MPI import UNDEFINED


# * INTERNAL LIBRARIES
from lib.solvers.base_solver           import BaseSolver
from lib.models.microvascular_network  import MicrovascularNetwork
from lib.models.direct_tone_regulation import DirectToneRegulation



#%% CLASS

class DirectNetworkTone(BaseSolver):

    models = ['MicrovascularNetwork', 'DirectToneRegulation']

    def configure_mpi(self, comm):

        super().configure_mpi(comm)

        # ? create a subcomm for the tone model
        if self.comm is not None:
            color = 0 if self.rank == 0 else UNDEFINED
            self.tcomm = self.comm.Split(color=color, key=0)        
        else:
            self.tcomm=comm


    def create_system(self):

        net_io = self.io.models['MicrovascularNetwork']
        ton_io = self.io.models['DirectToneRegulation']

        # # STORE PARAMETERS
        # * Time parameters
        self.t  = 0                               # initial time [s]
        self.dt = ton_io.solution['dt'][0]        # time step size [s]
        self.Nt = round( self.tend / self.dt )    # number of steps

        # ? flags for updating the network
        self._network_root = False
        self._network_tone = False

        # * Vaso-activity parameters
        self.groups = None # all vessels dilate

        # # CONFIGURE MICROVASCULAR NETWORK
        self.network = MicrovascularNetwork(
            net_io.network, 
            net_io.fluid, 
            net_io.inflow, 
            net_io.outflow, 
            self.comm
        )
        # owned by all ranks

        if self.rank == 0:
            output = self.io.output_dir / f'level_averaged_data/'
            makedirs(output, exist_ok=False)

        # # CONFIGURE TONE REGULATION
        if self.rank == 0:
            self.tone = DirectToneRegulation(
                self.dt,
                ton_io.contributions,
                self.tcomm
            )

        else:
            self.tone = None


        # # COUPLED PARAMETERS
        self._tension_curr = 0
        self._shear_curr   = 0
        self._lmbda_curr   = 1.0


    def initialise_solver(self):

        # ? solve network model in initial state
        self.update_root_condition(force=True)
        self.tension_curr, self.shear_curr = self.compute_network_state()

        # ? update reference values in tone model
        if self.tone is not None:
            self.tone.update_reference_tension(self.tension_curr)
            self.tone.update_reference_shear(self.shear_curr)

        # ? write initial condition
        self.m_attrs = self.get_model_dict(a=self)
        self.io.write_solutions(-1, 0, **self.m_attrs)


    def step(self, step):

        # # SOLVE MICRO-VESSEL NETWORK
        # ? update tone-dependent conditions
        self.update_network_tone()

        # ? update root boundary condition
        self.update_root_condition(force=self._network_tone)

        # ? solve
        self.tension_curr, self.shear_curr = self.compute_network_state() 
        # ! does this replace the getters and setters?

        # # SOLVE TONE REGULATION
        if self.tone is not None:

            # ? update current values for tone
            self.tone.update_current_tension(self.tension_curr)
            self.tone.update_current_shear(self.shear_curr)

            # ? solve
            self.tone.solve()


        # # UPDATE TIME
        self.t += self.dt


    def compute_network_state(self):

        if self._network_root or self._network_tone:

            self.update_terminal_condition()
            self.network.solve()          

            # ? compute coupling parameters
            self.network.compute_vessel_parameters()

            # ? reduce override counter
            self._network_root = False
            self._network_tone = False

            
        tension = self.network.compute_tension_average(
                self.groups
        )
        shear = self.network.compute_shear_average(
                self.groups
        )

        return tension, shear


    def update_root_condition(self, force=False):

        # # COMPUTE NEW ROOT PRESSURE
        new_root = self.rc.compute_value(self.t)

        # # COMPARE NEW PRESSURE AGAINST PREVIOUS NETWORK PRESSURE
        has_changed = not isclose(
            new_root,               # updated scale
            self.network.root,      # current scale
            rtol=0.0, 
            atol=1e-1
        )

        # # APPLY
        if has_changed or force:

            # print('updating root condition: ', new_root)

            # ? update root condition
            self.network.solver.update_inlet_condition( new_root )

            # ? increment override step counter
            self._network_root = True

            # print('updated!')


    def update_terminal_condition(self):
        self.network.solver.update_outlet_condition()


    def update_network_tone(self, force=False):

        # # COMPARE CURRENT SOLUTION FOR LAMBDA AGAINST PREVIOUS NETWORK LAMBDA
        # ? broadcast updated scale to other ranks on the network comm
        if ( self.comm is not None ) and ( self.comm.Get_size() > 1 ):
            new_lmbda = self.comm.bcast( 
                self.tone.lmbda if self.rank == 0 else None, 
                root=0 
            )
        else:
            new_lmbda  = self.tone.lmbda


        has_dilated = not isclose(
            new_lmbda,    # updated scale
            self.network.lmbda, # current scale
            rtol=0.0, 
            atol=1e-3
        )

        if has_dilated or force:

            self.lmbda_curr = new_lmbda

            # print('updating network tone: ', self.new_lmbda)

            # ? update tree scale
            self.network.solver.update_vessel_properties(
                self.lmbda_curr, groups=self.groups
            )

            # ? increment override step counter
            self._network_tone = True


    @property
    def tension_curr(self):
        return self._tension_curr

    @tension_curr.setter
    def tension_curr(self, value):
        self._tension_curr = value

    @property
    def shear_curr(self):
        return self._shear_curr

    @shear_curr.setter
    def shear_curr(self, value):
        self._shear_curr = value

    @property
    def lmbda_curr(self):
        return self._lmbda_curr

    @lmbda_curr.setter
    def lmbda_curr(self, value):
        self._lmbda_curr = value