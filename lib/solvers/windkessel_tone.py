"""
auth: L. J. Barratt
date: 28/09/26


"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
from numpy import max, abs, round
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

        # # CREATE WINDKESSEL SYSTEM
        # ? properties of the feed artery
        r0    = 0.0100
        Rn    = 2.07e7
        c0    = 6.1429

        fluid = self.extract_all_values(wk_io.fluid,   0)
        frctl = self.extract_all_values(wk_io.fractal, 0)
        self.wk = VariableWindkessel(
            wk_io.solution['dt'][0],    # time step
            self.rc.compute_value(0),   # initial pin
            r0, Rn, c0,                 # properties of feed artery
            fluid, frctl                # fluid & fractal properties
        )

        # # CREATE TONE REGULATION SYSTEM
        time_constants = self.extract_all_values(tone_io.time_constants, 0)
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
        self.dt = min( self.wk.dt, self.tone.dt ) # global time step
        self.Nt = int(round(self.tend/self.dt))+1 # number of steps

        # step increments
        self.step_wk   = self.wk.dt   / self.dt
        self.step_tone = self.tone.dt / self.dt


    def initialise_solver(
        self,
        maxiter=50,
        lsiter=30,
        tol=1e-6,
        called=False,
    ):
        '''
        Find the steady-state lambda per generation satisfying the coupled Windkessel and vaso-tone system.
        '''

        self.update_windkessel_inflow()

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

        if not called:
            self.m_attrs = self.get_model_dict(a=self)
            self.io.write_solutions(-1, 0, **self.m_attrs)


    def step(self, step):

        # # SOLVE DYNAMIC SYSTEMS
        # ? windkessel model
        if not step % self.step_wk:
            self.update_windkessel_inflow()

            self.wk.apply_dilation(self.tone.lmbda)
            self.wk.solve()
            

        # ? tone model
        if not step % self.step_tone:
            self.update_tone_state()
            self.tone.solve()

        # # UPDATE TIME
        self.t += self.dt
        

    def update_windkessel_inflow(self):
        new_root = self.rc.compute_value(self.t)
        self.wk.update_root_pressure( new_root )


    def update_tone_state(self):

        # get values from windkessel
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
