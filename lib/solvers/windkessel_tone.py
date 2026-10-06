"""
auth: L. J. Barratt
date: 28/09/26


"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
from numpy import max, abs, round
from math  import isclose
from sys   import exit

# * INTERNAL LIBRARIES
from lib.solvers.base_solver               import BaseSolver
from lib.models.variable_windkessel        import VariableWindkessel
from lib.models.vectorised_tone_regulation import VectorisedToneRegulation

from lib.etc.time_function import TimeFunction

# unit conversions
PA_PER_M_TO_DYN_PER_CM = 1e3
PA_TO_DYN_PER_CM2      = 10

#%% CLASS

class WindkesselTone(BaseSolver):

    models = ['VariableWindkessel', 'VectorisedToneRegulation']

    def configure_mpi(self, comm):

        super().configure_mpi(comm)

        if ( comm is not None ) and ( comm.Get_size() != 1 ):
            raise TypeError('WindkesselTone does not support MPI.')

    
    def create_system(self):

        wk_io   = self.io.model_io['VariableWindkessel']
        tone_io = self.io.model_io['VectorisedToneRegulation']
        
        tdt = tone_io.solution['dt'][0]

        # # CREATE WINDKESSEL SYSTEM
        # ? dt and properties of the feed artery
        if hasattr(self, 'extras'):
            # given by parent solver
            wkdt, r0, c0, Rn, ii = self.extras

        else:        
            # ! testing parameters
            r0, c0, Rn, ii = 0.0001, 6.1429, 2.07e7, 0
            wkdt = wk_io.solution.get('dt', [tdt])[0]

        # if wkdt exists, it must be equal to / smaller than tdt
        if wkdt > tdt:
            raise ValueError('wk dt must be equal or less than tone dt.')


        fluid = self.extract_all_keys(wk_io.fluid)
        frctl = self.extract_all_keys(wk_io.fractal)
        self.wk = VariableWindkessel(
            wkdt,               # time step
            r0, Rn, c0,         # properties of feed artery
            fluid, frctl,       # fluid & fractal properties
            wkid=ii
        )


        # # CREATE TONE REGULATION SYSTEM
        time_constants = self.extract_all_keys(tone_io.time_constants)
        self.tone = VectorisedToneRegulation(
            tone_io.solution['dt'][0],
            time_constants,
            self.wk,
        )

        # ? optional SNA term for tone model, 0 if not provided
        # term is equal across generations
        if tone_io.optional.sna is not None:
            self.sna = TimeFunction(tone_io.optional.sna)
        else:
            params   = { 'form': ['constant'], 'value': [0.0] }
            self.sna = TimeFunction(params)


        # # STORE PARAMETERS
        # * Time parameters
        self.t  = 0                               # initial time [s]
        self.dt = self.wk.dt                      # global time step
        self.Nt = int(round(self.tend/self.dt))+1 # number of steps

        # step increments for tone model
        ratio = self.tone.dt / self.dt
        if not isclose(ratio, round(ratio)):
            raise ValueError('tone dt must be an integer multiple of wk dt.')
        
        self.step_tone = int(round(ratio))
        self._reset_accumulators()


    def initialise_solver(
        self,
        maxiter=100,
        lsiter=100,
        tol=1e-6,
        called=False,
        pressure=None
    ):
        '''
        Find the steady-state lambda per generation satisfying the coupled Windkessel and vaso-tone system.
        '''

        self.update_windkessel_inflow(pressure)

        for itr in range(maxiter):

            lambda_old = self.tone.lmbda.copy()

            # ---------------------------------------------------------
            # Evaluate fixed-point map:
            #
            # lambda_old -> WK SS -> tone inputs -> tone SS
            #              -> lambda_target = G(lambda_old)
            # ---------------------------------------------------------

            self.wk.apply_dilation(lambda_old)
            self.wk.steady_state_solve()
            self.update_tone_state()
            self.tone.steady_state_solve()

            lambda_target = self.tone.lmbda.copy()

            # Fixed-point residual at current point
            F_old = lambda_target - lambda_old
            norm_F_old = max(abs(F_old))

            if norm_F_old < tol:
                break

            # ---------------------------------------------------------
            # Backtracking line search
            #
            # lambda_trial = lambda_old + alpha * F_old
            #
            # Accept only if ||F(lambda_trial)|| < ||F(lambda_old)||
            # ---------------------------------------------------------

            alpha = 1.0

            for lsitr in range(lsiter):

                lambda_trial = lambda_old + alpha * F_old

                # Evaluate G(lambda_trial)
                self.wk.apply_dilation(lambda_trial)
                self.wk.steady_state_solve()
                self.update_tone_state()
                self.tone.steady_state_solve()

                lambda_trial_target = self.tone.lmbda.copy()

                F_trial = lambda_trial_target - lambda_trial
                norm_F_trial = max(abs(F_trial))

                if norm_F_trial < norm_F_old:
                    break

                alpha *= 0.5

            else:
                raise ValueError(
                    f"line search failure at iteration {itr}({lsitr}): "
                    f"||F||={norm_F_old:.3e}"
                )

            # ---------------------------------------------------------
            # Accept lambda_trial as the next fixed-point iterate
            # ---------------------------------------------------------

            self.tone.lmbda = lambda_trial.copy()

            # Optional convergence test after accepted step
            if norm_F_trial < tol:
                break

        else:
            raise RuntimeError(
                f"no convergence in {maxiter} iterations: "
                f"res={norm_F_trial:.3e}"
            )

        self.tone.set_basal_state()

        # ? write initial solution
        if not called:
            self.m_attrs = self.get_model_dict(a=self)
            self.io.write_solutions(-1, 0, **self.m_attrs)


    def step(self, step, pressure=None):

        # # SOLVE WINDKESSEL SYSTEM (EVERY STEP)
        self.update_windkessel_inflow(pressure)
        self.wk.apply_dilation(self.tone.lmbda)
        self.wk.solve()

        # accumulate for time average
        k  = self.wk.Nmid
        st = self.wk.generation_state()
        self._T_sum   = self._T_sum   + st['tension'][k:]
        self._tau_sum = self._tau_sum + st['tau'][k:]
        self._nacc   += 1

        # # SOLVE DILATION & ACTIVATION (INCREMENTAL)
        # ! +1 so that this steps at the end of the wk time-window 
        # ! required for the time-averaging of the coupled parameters
        if not (step + 1) % self.step_tone:
            self.update_tone_state(
                self._T_sum / self._nacc,
                self._tau_sum / self._nacc
            )
            self.tone.solve()
            self._reset_accumulators()

        # # UPDATE TIME
        self.t += self.dt
        

    def update_windkessel_inflow(self, pressure=None):
        if pressure is None:
            pressure = self.rc.compute_value(self.t)

        self.wk.update_root_pressure(pressure)


    def update_tone_state(self, tension=None, shear=None):
        # get values from windkessel (if not given)
        if tension is None:
            k       = self.wk.Nmid
            st      = self.wk.generation_state()
            tension = st['tension'][k:]
            shear   = st['tau'][k:]

        # set values with unit conversion (SI to cgs)
        self.tone.T_cur    = tension * PA_PER_M_TO_DYN_PER_CM
        self.tone.myo_term = tension * PA_PER_M_TO_DYN_PER_CM
        self.tone.tau_term = shear   * PA_TO_DYN_PER_CM2

        # time-dependent sna term
        self.tone.csna = self.sna.compute_value(self.t)


    def _reset_accumulators(self):
        self._T_sum, self._tau_sum, self._nacc = 0.0, 0.0, 0


# # ############################ ################# ##########################
# # ############################ GETTERS & SETTERS ##########################
# # ############################ ################# ##########################


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
