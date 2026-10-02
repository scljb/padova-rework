"""
auth: L. J. Barratt
date: 02/09/26


"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
from os     import makedirs
import sys


# * INTERNAL LIBRARIES
from lib.solvers.direct_network_tone    import DirectNetworkTone
from lib.models.microvascular_network   import MicrovascularNetwork
from lib.models.tone_regulation         import ToneRegulation


#%% CLASS

class NetworkTone(DirectNetworkTone):

    models = ['MicrovascularNetwork', 'ToneRegulation']

    def create_system(self):

        net_io = self.io.models['MicrovascularNetwork']
        ton_io = self.io.models['ToneRegulation']

        # # STORE PARAMETERS
        # * Time parameters
        self.t  = 0                               # initial time [s]
        self.dt = ton_io.solution['dt'][0]        # time step size [s]
        self.Nt = round( self.tend / self.dt )    # number of steps

        # counter for step overrides of network
        self.step_override = 1

        # * Vaso-activity parameters
        self.groups = None # all vessels dilate

        # * Inflow condition
        self.y1         = net_io.inflow['value'][0]  # root pressure 1 [Pa]
        self.y2         = net_io.inflow['value2'][0] # root pressure 2 [Pa]
        self.delta      = net_io.inflow['delta'][0]  # time delay [s]
        self.tau        = net_io.inflow['tau'][0]    # time constant [s]   
        
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
            # only owned by rank 0
            self.tone = ToneRegulation(
                ton_io.diameter, 
                ton_io.activation,
                ton_io.solution,
                ton_io.contributions,
                self.tcomm
            )

        else:
            self.tone = None


        # # COUPLED PARAMETERS
        self.tension_curr = 0
        self.shear_curr   = 0
        self.lmbda_curr   = 1.0


    def initialise_solver(self, maxiter=100, tol=1e-6):

        for _ in range(maxiter):

            # ? previous values
            lmbda_old = self.lmbda_curr

            # ? solve network model in initial state
            self.update_root_condition()
            self.update_terminal_condition()
            self.network.solve()

            self.network.compute_vessel_parameters()
            tension = self.network.compute_tension_average(
                self.groups
            )
            shear = self.network.compute_shear_average(
                    self.groups
            )

            # print(tension, shear)

            # ? update reference values in tone model
            if self.tone is not None:
                self.tension_curr = tension * 1000
                self.shear_curr   = shear   * 10
                self.tone.update_reference_tension(self.tension_curr)
                self.tone.update_current_tension(self.tension_curr)
                self.tone.update_current_shear(self.shear_curr)
                self.tone.initial_solve()
                dx = self.tone.lmbda - lmbda_old

                if _ == 0:
                    self.tone.lmbda += dx

                k_tone = self.tone.compute_muscle_tone()
                print(f'{_} \t\t {self.tone.lmbda:.4f} \t\t {dx:.4f}')
                # print(self.tension_curr, self.shear_curr)

            self.update_network_tone()


            err = abs( self.lmbda_curr - lmbda_old )
            # if err < tol:
            #     print('yay')
            #     break


        if err > tol:
            raise RuntimeError('nooooooooo')


        # ? reset state and update ref state
        if self.rank == 0:
            print(self.tone.lmbda, self.network.lmbda, self.lmbda_curr)

        self.update_network_reference_state() # ! update vessels list
        self.step_override = 1
        self.lmbda_curr    = 1.0

        if self.tone is not None:
            self.tone.lmbda = 1.0
            self.tone.k_tone = -k_tone
            self.tone.update_reference_tension(self.tension_curr*1000)

            print(self.tone.compute_muscle_tone())


        # ? write initial condition
        self.m_attrs = self.get_model_dict(a=self)
        self.io.write_solutions(-1, 0, **self.m_attrs)


    def step(self, step):

        # # SOLVE MICRO-VESSEL NETWORK
        # ? update tone-dependent network
        self.update_network_tone()

        # ? update time-dependent conditions
        self.update_root_condition()
        self.update_terminal_condition()

        # ? solve
        if self.step_override > 0:

            self.network.solve()          

            # ? compute coupling parameters
            self.network.compute_vessel_parameters()

            # ? reduce override counter
            self.step_override -= 1

            
        tension = self.network.compute_tension_average(
                self.groups
        )
        shear = self.network.compute_shear_average(
                self.groups
        )

        # # SOLVE TONE REGULATION
        if self.tone is not None:

            self.tension_curr = tension * 1000
            self.shear_curr   = shear   * 10

            # ? update current values for tone
            self.tone.update_current_tension(self.tension_curr)
            self.tone.update_current_shear(self.shear_curr)

            # ? solve
            self.tone.solve()


        # # UPDATE TIME
        self.t += self.dt


    def update_network_reference_state(self):
        for ves in self.network.vessels:
            ves.r0 = ves.r
            ves.c0 = ves.C0 * self.network.lmbda

        self.network.lmbda = 1.0