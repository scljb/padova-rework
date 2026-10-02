"""
auth: L. J. Barratt
date: 27/09/26


"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
from numpy      import isclose
from os         import makedirs
import sys

# * INTERNAL LIBRARIES
from lib.solvers.base_solver           import BaseSolver
from lib.models.windkessel             import Windkessel
from lib.models.microvascular_network  import MicrovascularNetwork
from lib.models.direct_tone_regulation import DirectToneRegulation

#%% CLASS

class DirectMicrovascularTone(BaseSolver):

    models = ['Windkessel', 'MicrovascularNetwork', 'DirectToneRegulation']
    
    def create_system(self):

        wk_io   = self.io.model_io['Windkessel']
        net_io  = self.io.model_io['MicrovascularNetwork']
        tone_io = self.io.model_io['DirectToneRegulation']

        # # STORE PARAMETERS
        # * Time parameters
        self.t  = 0                               # initial time [s]
        self.dt = tone_io.solution['dt'][0]       # time step size [s]
        self.Nt = round( self.tend / self.dt )    # number of steps

        # # CREATE WINDKESSEL SYSTEM
        # ? vascular properties
        vkeys = [ 'r1', 'r2', 'c' ]
        vasc  = self.extract_values(wk_io.vascular, vkeys, 0)

        # ? fluid properties
        fkeys = [ 'capacitor_pressure', 'outflow_pressure' ]
        fluid = self.extract_values(wk_io.fluid, fkeys, 0)

        # ? instance class on lead rank
        self.wk = None
        if self.rank == 0:
            self.wk = Windkessel(
                self.dt, 
                0,
                vasc,
                fluid,
            )

        # # CREATE FRACTAL SYSTEM
        # ? flags for updating the network
        self.ROOT_FLAG = False
        self.TONE_FLAG = False

        # ? instance class
        self.network = MicrovascularNetwork(
            net_io.network, 
            net_io.fluid, 
            self.comm
        )

        if self.rank == 0:
            output = self.io.output_dir / f'level_averaged_data/'
            makedirs(output, exist_ok=False)


        # # CREATE TONE REGULATION SYSTEM
        # ? instance class on lead rank
        self.tone = None
        if self.rank == 0:
            self.tone = DirectToneRegulation(
                self.dt,
                tone_io.contributions,
            )

        # # COUPLED PARAMETERS
        self.network_tension = None
        self.network_shear   = None
        self.network_rtot    = None
        self.network_ceff    = None


    def initialise_solver(self, called=False):
        '''
        At steady state:
            - p_in == constant
            - d[pcap]/dt  == 0
            - d[lmbda]/dt == 0
        '''

        # ? find steady state for windkessel
        if self.rank == 0:
            self.update_windkessel_inflow()
            self.wk.steady_state_solve()

        # ? solve fractal model in steady state
        self.update_fractal_inflow(force=True)
        self.update_fractal_state(force=True)

        # ? find steady state for tone
        if self.rank == 0:
            self.update_tone_state()
            self.tone.steady_state_solve()

        # if self.rank == 0:
        #     print(f'{len(self.network.vessels)}', f'{self.network_rtot:.6e}', f'{self.network_ceff:.6e}')

        # ? write initial condition
        if not called:
            self.m_attrs = self.get_model_dict(a=self)
            self.io.write_solutions(-1, 0, **self.m_attrs)


    def step(self, step):

        # # SOLVE DYNAMIC SYSTEMS
        if self.rank == 0:

            # ? windkessel model
            self.update_windkessel_inflow()
            self.update_windkessel_state()
            self.wk.solve()

            # ? tone model
            self.update_tone_state()
            self.tone.solve()

        # # UPDATE STATIC FRACTAL
        self.update_fractal_tone()
        self.update_fractal_inflow( force=self.TONE_FLAG )

        if self.rank == 0:
            if not step % 10: 
                print(f'{self.t:.6f}\t',
                    f'{self.wk.pin:.6f}\t'
                    f'{self.wk.pcap:.6f}\t'
                    f'{self.tone.lmbda:.6f}\t', 
                    f'{self.ROOT_FLAG}\t',
                    f'{self.TONE_FLAG}\t'
                    f'{self.network_tension:.6f}\t', 
                    f'{self.network_shear:.6f}\t', 
                    f'{self.network_rtot:.6e}\t', 
                    f'{self.network_ceff:.6e}\t')

        self.update_fractal_state(  force=( self.ROOT_FLAG or self.TONE_FLAG ) )

        # # UPDATE TIME
        self.t += self.dt
        

    def update_windkessel_inflow(self):
        new_root = self.rc.compute_value(self.t)
        self.wk.update_root_pressure( new_root )


    def update_windkessel_state(self):
        self.wk.R2 = self.network_rtot
        self.wk.C  = self.network_ceff


    def update_fractal_inflow(self, force=False):

        # ? broadcast updated windkessel p_c to other ranks on the global comm
        if ( self.comm is not None ) and ( self.comm.Get_size() > 1 ):
            pcap = self.comm.bcast( 
                self.wk.pcap if self.rank == 0 else None, 
                root=0 
            )
        else:
            pcap  = self.wk.pcap

        # ? compare against current network root pressure
        has_changed = False
        if not force:
            has_changed = not isclose(
                pcap,                   # updated scale
                self.network.root,      # current scale
                rtol=0.0, 
                atol=5e-1
            )

        # ? update if the difference surpasses the tolerance
        if has_changed or force:

            # ? update root condition
            self.network.solver.update_inlet_condition( pcap )

            # ? set step flag due to root changes
            self.ROOT_FLAG = True


    def update_fractal_state(self, force:bool):

        if not force:
            return


        self.network.solver.update_outlet_condition(0.0)
        self.network.solve()          

        # ? compute coupling parameters
        self.network.compute_vessel_parameters()

        # ? reduce override counter
        self.ROOT_FLAG = False
        self.TONE_FLAG = False

        # ? update network coupled parameters
        self.network_tension = self.network.compute_tension_average()
        self.network_shear   = self.network.compute_shear_average()
        self.network_rtot    = self.network.compute_tree_resistance()
        self.network_ceff    = self.network.compute_tree_compliance()


    def update_fractal_tone(self, force=False):

        # ? broadcast updated lambda to other ranks on the global comm
        if ( self.comm is not None ) and ( self.comm.Get_size() > 1 ):
            new_lmbda = self.comm.bcast( 
                self.tone.lmbda if self.rank == 0 else None, 
                root=0 
            )
        else:
            new_lmbda  = self.tone.lmbda


        # ? compare against current network scale
        has_changed = False
        if not force:
            has_changed = not isclose(
                new_lmbda,    # updated scale
                self.network.lmbda, # current scale
                rtol=0.0, 
                atol=5e-4
        )

        # ? update if the difference surpasses the tolerance
        if has_changed or force:

            # ? update tree scale
            self.network.solver.update_vessel_properties(new_lmbda)

            # ? set step flag due to tone changes
            self.TONE_FLAG = True


    def update_tone_state(self):
        self.tone.myo_term = self.network_tension
        self.tone.tau_term = self.network_shear


    @property
    def ROOT_FLAG(self):
        return self._ROOT_FLAG

    @ROOT_FLAG.setter
    def ROOT_FLAG(self, value):
        self._ROOT_FLAG = value


    @property
    def TONE_FLAG(self):
        return self._TONE_FLAG

    @TONE_FLAG.setter
    def TONE_FLAG(self, value):
        self._TONE_FLAG = value


    @property
    def network_tension(self):
        return self._network_tension

    @network_tension.setter
    def network_tension(self, value):
        self._network_tension = value


    @property
    def shear_tension(self):
        return self._shear_tension

    @shear_tension.setter
    def shear_tension(self, value):
        self._shear_tension = value


    @property
    def rtot_tension(self):
        return self._rtot_tension

    @rtot_tension.setter
    def rtot_tension(self, value):
        self._rtot_tension = value


    @property
    def ceff_tension(self):
        return self._ceff_tension

    @ceff_tension.setter
    def ceff_tension(self, value):
        self._ceff_tension = value