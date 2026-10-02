"""
auth: L. J. Barratt
date: 28/09/26


"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
from numpy  import isclose, sqrt, pi
from os     import makedirs
from sys    import exit
import traceback

# * INTERNAL LIBRARIES
from lib.solvers.base_solver           import BaseSolver
from lib.models.windkessel             import Windkessel
from lib.models.microvascular_network  import MicrovascularNetwork
from lib.models.tone_regulation        import ToneRegulation

#%% CLASS

class MicrovascularTone(BaseSolver):

    models = ['Windkessel', 'MicrovascularNetwork', 'ToneRegulation']
    
    def create_system(self):

        wk_io   = self.io.model_io['Windkessel']
        net_io  = self.io.model_io['MicrovascularNetwork']
        tone_io = self.io.model_io['ToneRegulation']

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
                wk_io.solution['dt'][0], 
                self.rc.compute_value(0),
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

        # todo idea:
        # probe which vessel groups are present in network
        # 


        # # CREATE TONE REGULATION SYSTEM
        # ? instance class on lead rank
        self.tone = None
        if self.rank == 0:
            self.tone = ToneRegulation(
                tone_io.solution['dt'][0],
                tone_io.diameter,
                tone_io.activation,
                tone_io.contributions,
            )

        # # STORE PARAMETERS
        # * Time parameters
        if ( self.comm is not None ) and ( self.comm.Get_size() > 1 ):
            wk_dt = self.comm.bcast( 
                self.wk.dt if self.wk is not None else None, 
                root=0 
            )            
            tone_dt = self.comm.bcast( 
                self.tone.dt if self.tone is not None else None, 
                root=0 
            )
        else:
            wk_dt   = self.wk.dt
            tone_dt = self.tone.dt


        self.t  = 0                              # initial time [s]
        self.dt = min( wk_dt, tone_dt )          # global time step
        self.Nt = round( self.tend / self.dt )+1 # number of steps

        # step increments
        self.step_wk   = wk_dt   / self.dt
        self.step_tone = tone_dt / self.dt

        # # COUPLED PARAMETERS
        self.network_tension = None
        self.network_shear   = None
        self.network_rtot    = None
        self.network_ceff    = None


    def initialise_solver(
            self, 
            max_outer=50, 
            max_inner=50, 
            tol=1e-8, 
            alpha=1.0, 
            called=False
    ):
        '''
        Find the scalar lmbda where the coupled (WK + HP network + tone) system is at steady state for the initial root pressure, then make that the reference state.
        '''

        # ? set initial state
        if self.rank == 0:
            self.update_windkessel_inflow()
            self.tone.lmbda = 1.0

        
        # ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.rank == 0:
            print(
                f'{'itr':6}\t',
                f'{'lmbda':12}\t',
                f'{'pcap':12}\t',
                f'{'r0':12}\t',
                f'{'tension':12}\t', 
                f'{'t_total':12}\t', 
                f'{'shear':12}\t', 
                f'{'rtot':12}\t', 
                f'{'ceff':12}\t',
                f'{'ctone':12}\t'
            )
        # ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        for itr in range(max_outer):

            # # INNER LOOP
            for _ in range(max_inner):
                self.update_fractal_inflow(force=True) # applies pcap
                self.update_fractal_state(force=True)  # T, tau, R2, C

                dpc, fail = None, None
                if self.rank == 0:
                    try:
                        pcap_old = self.wk.pcap
                        self.update_windkessel_state() # applies R2, C
                        self.wk.steady_state_solve()
                        dpc = abs(self.wk.pcap - pcap_old) / max(abs(pcap_old), 1e-12)
                    except Exception:
                        fail = traceback.format_exc()

                dpc, fail = self.comm.bcast((dpc, fail), root=0)
                if fail is not None:
                    raise RuntimeError(f'windkessel failed:\n{fail}')
                
                if dpc < tol:
                    break

            else:
                raise RuntimeError(
                    f'inner pcap loop did not converge, dpc={dpc:.2e}'
                )

            # ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if self.rank == 0:
                tt = self.tone._Tpassive(self.tone.lmbda) + ( self.tone.theta * self.tone._Tactive(self.tone.lmbda) )

                print(f'{itr:6}\t',
                    f'{self.tone.lmbda:6.6f}\t'
                    f'{self.wk.pcap:6.6f}\t'
                    f'{self.network.vessels[0].r0:6.6f}\t'
                    f'{self.network_tension:6.6f}\t', 
                    f'{tt:6.6f}\t', 
                    f'{self.network_shear:6.6f}\t', 
                    f'{self.network_rtot:6.6e}\t', 
                    f'{self.network_ceff:6.6e}\t',
                    f'{self.tone.c_tone:6.6e}\t', 
                )
            # ! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            # # TONE SOLVE
            res, fail = None, None
            if self.rank == 0:
                try:
                    self.update_tone_state()       # applies T, tau
                    self.tone.steady_state_solve() # lmbda, c_tone, T_ref
                    res = abs(self.tone.lmbda - 1.0)

                except Exception:
                    fail = traceback.format_exc()

            res, fail = self.comm.bcast((res, fail), root=0)
            if fail is not None:
                raise RuntimeError(f'tone solve failed:\n{fail}')

            if res < 1e-6:
                break

            # # APPLY lmbda to NETWORK, UPDATE REFERENCE STATE
            if self.rank == 0:
                dlmbda = self.tone.lmbda - 1.0
                self.tone.lmbda = 1.0 + alpha*dlmbda

            self.update_fractal_tone(force=True)   # r = lmbda * r0
            self.update_fractal_reference()        # resets r0=r, C0=C
            if self.rank == 0:
                self.tone.lmbda = 1.0              # lmbda = 1.0

        else:
            raise RuntimeError(f'no convergence, |lmbda-1|={res:.2e}')


        # ? write initial condition
        if not called:
            self.m_attrs = self.get_model_dict(a=self)
            self.io.write_solutions(-1, 0, **self.m_attrs)


    def step(self, step):

        # # SOLVE DYNAMIC SYSTEMS
        if self.rank == 0:

            # ? windkessel model
            if not step % self.step_wk:
                self.update_windkessel_inflow()
                self.update_windkessel_state()
                self.wk.solve()

            # ? tone model
            if not step % self.step_tone:
                self.update_tone_state()
                self.tone.solve()


        # # UPDATE STATIC FRACTAL
        self.update_fractal_tone()
        self.update_fractal_inflow( force=self.TONE_FLAG )

        # if self.rank == 0:
        #     if not step % 10: 
        #         print(f'{self.t:.2f}\t',
        #             f'{self.wk.pin:7.6f}\t'
        #             f'{self.wk.pcap:7.6f}\t'
        #             f'{self.tone.lmbda:7.6f}\t', 
        #             f'{self.ROOT_FLAG}\t',
        #             f'{self.TONE_FLAG}\t'
        #             f'{self.network_tension:7.6f}\t', 
        #             f'{self.network_shear:7.6f}\t', 
        #             f'{self.network_rtot:.6e}\t', 
        #             f'{self.network_ceff:.6e}\t'
        #         )

        self.update_fractal_state( force=( self.ROOT_FLAG or self.TONE_FLAG ) )

        # # UPDATE TIME
        self.t += self.dt
        

    def update_windkessel_inflow(self):
        new_root = self.rc.compute_value(self.t)
        self.wk.update_root_pressure( new_root )


    def update_windkessel_state(self):
        # todo add damping to updates of the WK

        self.wk.R2 = self.network_rtot
        self.wk.C  = self.network_ceff


    def update_fractal_inflow(self, force=False):

        # ? broadcast updated windkessel pcap to other ranks on the global comm
        if ( self.comm is not None ) and ( self.comm.Get_size() > 1 ):
            pcap = self.comm.bcast( 
                self.wk.pcap if self.wk is not None else None, 
                root=0 
            )
        else:
            pcap = self.wk.pcap

        # ? compare against current network root pressure
        has_changed = False
        if not force:
            has_changed = not isclose(
                pcap,                   # updated scale
                self.network.root,      # current scale
                rtol=0.0, 
                atol=1e-1
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

        # ? solve network
        self.network.solver.update_outlet_condition(0.0)
        self.network.solve()          

        # ? compute coupling parameters
        self.network.compute_vessel_parameters()

        # ? update network coupled parameters
        tension = self.network.compute_tension_average()
        shear   = self.network.compute_shear_average()
        self.network_rtot    = self.network.compute_tree_resistance()
        self.network_ceff    = self.network.compute_tree_compliance()

        # unit conversion
        if self.rank == 0:
            self.network_tension = tension * 1000
            self.network_shear   = shear   * 10

        # ? set flags to false
        self.ROOT_FLAG = False
        self.TONE_FLAG = False

        
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
                atol=1e-4
        )

        # ? update if the difference surpasses the tolerance
        if has_changed or force:

            # ? update tree scale
            self.network.solver.update_vessel_properties(new_lmbda)

            # ? set step flag due to tone changes
            self.TONE_FLAG = True


    def update_fractal_reference(self):
        '''
        Called during intialisation to update the reference state of the fractal.
        '''

        # ? update each vessel
        for ves in self.network.vessels:
            ves.r0 = ves.r
            Eh     = self.network.compute_vessel_stiffness(ves.r0)
            beta   = 4*Eh*sqrt(pi) / 3
            A0     = pi*ves.r0**2
            ves.C0 = 2.0*(A0**1.5) * ves.L / beta
            ves.C  = ves.C0



        # ? update root radii 
        self.network.r0 *= self.network.lmbda

        # ? reset lmbda=r/r0 to 1.0
        self.network.lmbda = 1.0


    def update_tone_state(self):
        self.tone.T_cur    = self.network_tension
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

