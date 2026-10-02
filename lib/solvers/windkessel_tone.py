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
            self.wk
        )

        # # STORE PARAMETERS
        # * Time parameters
        self.t  = 0                               # initial time [s]
        self.dt = min( self.wk.dt, self.tone.dt ) # global time step
        self.Nt = int(round(self.tend/self.dt))+1 # number of steps

        # step increments
        self.step_wk   = self.wk.dt   / self.dt
        self.step_tone = self.tone.dt / self.dt

        print(f'Rn={self.wk.Rn}, C={self.wk.C}')


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






















    def old_init_solver(self, maxiter=50, tol=1e-8, alpha=1.0, called=False):
        '''
        Find the scalar lmbda where the coupled (WK + tone) system is at steady
        state for the initial root pressure, then make that the reference state.
        '''

        # ? set initial state
        self.update_windkessel_inflow()
        self.tone.lmbda = 1.0

        # print(
        #     f'{'itr':6}\t',
        #     f'{'Rn':12}\t', 
        #     f'{'Rm':12}\t', 
        #     f'{'Rf':12}\t', 
        #     f'{'RT':12}\t'
        # )

        # print(
        #     f'{-1:5}',
        #     f'{self.wk.Rn:.6e}', 
        #     f'{self.wk.Rm:.6e}', 
        #     f'{self.wk.Rf:.6e}', 
        #     f'{self.wk.RT:.6e}'
        # )

        # exit()

        st = self.wk.generation_state()
        k = self.wk.Nmid

        print(
            f'Radius per generation: {st['radius'][k:]*1e6}\n'
            f'Vessels per generation: {2**st['gen'][k:]}'
        )

        # # LOOP OVER COUPLED SYSTEM
        for itr in range(maxiter):

            # ? windkessel at steady state
            self.wk.steady_state_solve()

            # ? tone at steady state
            self.update_tone_state()        # T, tau in far compartment of WK
            
            # ! debugging            
            print(
                f'T_cur / T_wall(lmbda=1) per generation: {self.tone.reference_ratio()}'
            )
            # ! gniggubed

            self.tone.steady_state_solve()  # lmbda, c_tone, T_ref

            # ? check convergence
            res = abs(self.tone.lmbda - 1.0)
            if res.max() < tol:
                break

            # ? apply lmbda to the far compartment
            lmbda = 1.0 + alpha * (self.tone.lmbda - 1.0)
            self.wk.apply_dilation(lmbda)

            st = self.wk.generation_state()
            k  = self.wk.Nmid
            # fun = lambda x: (
            #     self.tone._Tpassive(x) + self.tone.theta * self.tone._Tactive(x)
            # )
            # tt = fun(lmbda)

            # print(
            #     f'{itr:5}',
            #     f'{st['tension'][k:]}',
            # )



            # print(
            #     f'{itr:5}\t',
            #     f'{self.wk.Rn:.6e}\t', 
            #     f'{self.wk.Rm:.6e}\t', 
            #     f'{self.wk.Rf:.6e}\t', 
            #     f'{self.wk.RT:.6e}\t'
            # )

            # ? set new reference state
            self.wk.set_reference()
            self.tone.set_reference(self.wk)


        else:
            raise RuntimeError(f'no convergence, |lmbda-1|={res}')
            

        # ? write initial condition
        if not called:
            self.m_attrs = self.get_model_dict(a=self)
            self.io.write_solutions(-1, 0, **self.m_attrs)



def init_solver(self, maxiter=50, tol=1e-8, alpha=0.5):
        '''
        Find lmbda of each generation  

        Find the scalar lmbda where the coupled (WK + tone) system is at steady
        state for the initial root pressure, then make that the reference state.
        '''

        # ? set initial state
        self.update_windkessel_inflow()
        self.tone.lmbda = 1.0


        # # LOOP OVER COUPLED SYSTEM
        for itr in range(maxiter):

            lmbda = self.tone.lmbda.copy()

            # ? windkessel at steady state, solve for pcap
            self.wk.steady_state_solve()

            # ? update the current T, tau in the tone model
            self.update_tone_state()

            # ? tone at steady state, solve for lmbda
            self.tone.steady_state_solve()

            # ? check convergence of lmbda
            res = abs(self.tone.lmbda - 1.0)
            if res.max() < tol:
                break

            # ? dampened solution
            lmbda = 1.0 + alpha * (self.tone.lmbda - 1.0)

            # ? apply lmbda to the far compartment
            self.wk.apply_dilation(lmbda)# sollmbda, c_tone, T_ref

        else:
            raise RuntimeError(f'no convergence, |lmbda-1|={res}')


        # ? set new reference state in wk
        self.wk.set_reference()
        self.tone.set_reference()


def Old_initialise_solver(
        self, 
        maxiter=2000, 
        tol=1e-6, 
        alpha=0.000001, 
        called=False
    ):
    '''
    Find the scalar lmbda, for each generation, where the coupled (WK + tone) system is at steady state for the initial root pressure. Then, set the basal state in the tone model.
    '''

    # ? set initial state
    self.update_windkessel_inflow()

    # # SOLVE FOR STEADY STATE
    for itr in range(maxiter):

        Rf_old    = self.wk.Rf
        lmbda_old = self.tone.lmbda.copy()

        # ? windkessel at steady state, solve for pcap
        self.wk.steady_state_solve()

        # ? update the current T, tau in the tone model
        self.update_tone_state()

        # ? tone at steady state, solve for lmbda
        self.tone.steady_state_solve()
        lmbda = self.tone.lmbda.copy()

        # ? apply under-relaxed lmbda to WK tree
        lmbda_update = (1-alpha)*lmbda_old + alpha*lmbda
        self.wk.apply_dilation(lmbda_update)

        # * lambda residual
        res_lmbda = max(abs(lmbda_update - lmbda_old))

        # * Rf residual
        Rf = self.wk.Rf
        res_Rf = abs(Rf - Rf_old) / max([abs(Rf_old), 1e-30])

        if res_lmbda < tol and res_Rf < tol:
            break

        # ! debugging
        print(
            itr,
            f"lambda_eq =", round(lmbda, 6),
            f"lambda_wk =", round(lmbda_update, 6),
            f"res lmbda = {res_lmbda:.6f}",
            f"res Rf = {res_Rf:.6f}",
            # f"Rf = {self.wk.Rf}"
        )
        # ! ginggubed

    else:
        raise RuntimeError(f'no convergence in {itr} iterations, res={res_lmbda}, {res_Rf}')
    
    
    # # SET THE NEW REFERENCE STATE
    self.tone.set_basal_state()

    # ? write initial condition
    if not called:
        self.m_attrs = self.get_model_dict(a=self)
        self.io.write_solutions(-1, 0, **self.m_attrs)

    print("END INIT")
    print("tone:", self.tone.lmbda)
    print("wk:  ", self.wk.lmbda)
    print("diff:", self.wk.lmbda - self.tone.lmbda)