"""
auth: L. J. Barratt
date: 02/09/26
"""

#%% PACKAGES
from numpy           import exp, linspace, where, argmin, abs, asarray, ones_like
from scipy.optimize  import brentq
from scipy.integrate import solve_ivp
from scipy.special   import expit


#%% CLASS
class ToneRegulation:

    def __init__(self, dt, dia, act, con):

        # # SOLUTION PARAMETERS
        self.dt = dt

        # # TIME CONSTANTS
        self.tau_lmbda = dia['tau_lmbda'][0]
        self.tau_theta = act['tau_theta'][0]

        # # TENSION COEFFICIENTS
        # Passive tension
        self.cpas1 = dia['passive_1'][0]
        self.cpas2 = dia['passive_2'][0]

        # Active tension
        self.cact1 = dia['active_1'][0]
        self.cact2 = dia['active_1'][0]
        self.cact3 = dia['active_1'][0]

        # ? reference tension (set during steady state)
        self.T_ref = 0

        # ? current tension in micro-vessels
        self.T_cur = 0

        # # CONTRIBUTIONS TO TONE
        self.k_myo = con.get('myogenic',  [0])[0]
        self.k_tau = con.get('shear',     [0])[0]
        self.k_met = con.get('metabolic', [0])[0]

        # ? basal tone in the reference state
        self.c_tone = 0

        # intital values
        self.myo_term = 0
        self.tau_term = 0
        self.met_term = 0

        # # INITIAL CONDITION
        self.lmbda = 1.0
        self.theta = 0.5

        # # NUMERICAL METHOD
        # self.solve = self._forward_euler_solve
        self.solve = self._ivp_solve


    def steady_state_solve(self, tol=1e-12):
        '''
        At steady state,
            - d[theta]/dt == 0
            - d[lmbda]/dt == 0
            - S_tone      == 0
            - theta_tot   == 0.5
            - theta       == 0.5

        '''

        # ? steady state activation
        theta_ss = 0.5

        # ? residual function for steady state lmbda
        residual = lambda x: self.T_cur - ( 
            self._Tpassive(x) + theta_ss*self._Tactive(x) 
        )

        # ? solve across a range of physiological lmbda values
        lmbda_range = (0.2, 5.0)
        grid = linspace(*lmbda_range, 50)
        res  = residual(grid)

        # ? 
        idx  = where( res[:-1]*res[1:] <= 0 )[0]
        if len(idx) == 0:
            raise RuntimeError('No root for lmbda found.')

        # ? find lmbda (near 1.0) that produces steady state for current tension
        i = idx[ argmin( abs( grid[idx] - 1.0 ) ) ]
        lmbda_ss = brentq(residual, grid[i], grid[i+1], xtol=tol)

        # ? set reference values
        self.c_tone = self.tau_term + self.met_term - self.myo_term
        self.T_ref  = self.T_cur

        # ? update solution attributes
        self.lmbda = lmbda_ss
        self.theta = theta_ss       


    
    def _forward_euler_solve(self, t):
        ''' 
        explicit coupling for theta and lmbda
        '''

        # # COMPUTE ACTIVATION GRADIENT
        # ? smooth muscle tone
        S_tone = self.myo_term - self.tau_term - self.met_term + self.c_tone

        # ? activation solve
        dtheta = ( expit(S_tone) - self.theta ) / self.tau_theta

        # # COMPUTE DILATION SCALING GRADIENT
        # ? total wall tension that is counter-acting the fluid tension
        passive_tension = self._Tpassive(self.lmbda)
        active_tension  = self._Tactive(self.lmbda)
        total_tension   = passive_tension + ( self.theta * active_tension )

        # ? diameter solve
        dlmbda = ( self.T_cur - total_tension ) / ( 
                self.tau_lmbda * self.T_ref 
        )

        # # UPDATE OLD SOLUTION
        self.lmbda += self.dt * dlmbda
        self.theta += self.dt * dtheta


    def _ivp_solve(self):

        # # COMPUTE SOLUTIONS
        y_old = [self.lmbda, self.theta]
        sol = solve_ivp(
            lambda t, y: self._rates(*y), 
            (0, self.dt), 
            y_old,
            method='Radau', 
            rtol=1e-6, 
            atol=1e-9
        )
        self.lmbda, self.theta = sol.y[:, -1]


    def _rates(self, lmbda, theta):

        # ? smooth muscle tone
        S_tone = self.myo_term - self.tau_term - self.met_term + self.c_tone

        # ? activation ode
        dtheta = ( expit(S_tone) - theta ) / self.tau_theta

        # ? total wall tension that is counter-acting the fluid tension
        passive_tension = self._Tpassive(lmbda)
        active_tension  = self._Tactive(lmbda)
        total_tension   = passive_tension + ( theta * active_tension )

        # ? diameter ode
        dlmbda = ( self.T_cur - total_tension ) / ( 
                self.tau_lmbda * self.T_ref 
        )

        return dlmbda, dtheta


    def _Tpassive(self, lmbda):
        return self.cpas1 * exp( self.cpas2*( lmbda-1 ) )


    def _Tactive(self, lmbda):
        return self.cact1 * exp( 
            - ( ( lmbda - self.cact2 ) / self.cact3 )**2 
        )
    

    def _compute_tension_parameters(self, r):
        '''
        d : vessel diameter [um]. Valid range ~41-173 um.
        Sets cpas1, cpas2, cact1, cact2, cact3 according to diameter.
        '''

        d = 2*r
        self.cpas1 = -0.08507 + 6.666249*d  # linear, R2=1.000000
        self.cpas2 = 12.51775 - 0.026982*d  # linear, R2=0.999999
        self.cact1 = 1.30007  * d**1.47995  # power law, R2=0.999999
        self.cact2 = 0.77296  - 0.000590*d  # linear, R2=1.000000
        self.cact3 = 0.41499  - 0.000800*d  # linear, R2=1.000000


    @property
    def T_cur(self):
        return self._T_cur

    @T_cur.setter
    def T_cur(self, tension):
        self._T_cur = tension


    @property
    def myo_term(self):
        return self._myo_term

    @myo_term.setter
    def myo_term(self, tension):
        self._myo_term = self.k_myo * tension


    @property
    def myo_term(self):
        return self._myo_term

    @myo_term.setter
    def myo_term(self, tension):
        self._myo_term = self.k_myo * tension


    @property
    def tau_term(self):
        return self._tau_term

    @tau_term.setter
    def tau_term(self, shear):
        self._tau_term = self.k_tau * shear


    @property
    def met_term(self):
        return self._met_term

    @met_term.setter
    def met_term(self, oxygen):
        self._met_term = self.k_met * oxygen










































    
    def old_solve(self):

        # # COMPUTE EQUILIBRIUM ACTIVATION
        theta_total = 1 / ( 1 + exp( self.S_tone ) )

        # # SOLVE ACTIVATION ODE
        theta =  self.thetaold + (
            ( self.dt / self.tau_theta ) * ( theta_total - self.thetaold )
        )

        # # COMPUTE EQUILIBRIUM WALL TENSION
        # ? passive wall tension
        passive = (
            self.cpas1 
          * exp( self.cpas2 * ( self.lmbdaold - 1 ) ) 
        )

        # ? active wall tension
        active  = (
            self.cact1 
          * exp( - ( ( self.lmbdaold - self.cact2 ) / self.cact3 )**2 )
        )

        # ? total wall tension
        total_tension = passive + ( theta * active )

        # # SOLVE DIAMETER ODE
        lmbda =  self.lmbdaold + (
            ( self.dt / self.tau_lmbda ) * ( self.tension - total_tension )
        )

        # # UPDATE OLD SOLUTIONS
        self.lmbdaold = lmbda
        self.thetaold = theta

