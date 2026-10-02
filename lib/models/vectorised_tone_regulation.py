"""
auth: L. J. Barratt
date: 02/09/26
"""

#%% PACKAGES
from numpy import exp, abs, inf, ceil, log2, ones, zeros
from numpy import linspace, where, argmin, asarray, broadcast_to, \
                  concatenate, full

from scipy.integrate import solve_ivp
from scipy.special   import expit

# pressure at which the passive diameter Dp100 is defined [Pa]
P100 = 100.0 * 133.322

#%% CLASS
class VectorisedToneRegulation:

    def __init__(self, dt, time_constants, network):

        # # SOLUTION PARAMETERS
        self.dt = dt

        # # TIME CONSTANTS
        self.tau_lmbda = time_constants['tau_lmbda']
        self.tau_theta = time_constants['tau_theta']

        # # NETWORK PROPERTIES
        k        = network.Nmid
        self.gen = network.gen[k:]            # generation numbers
        self.n   = self.gen.size              # number of regulated generations
        r_um     = network.radius0[k:] * 1e6  # [m] -> [um], reference radii

        # # COEFFICIENTS
        self.cmet = 0.0
        self.update_coefficients(r_um)

        # ? tone and tension in the basal state
        # set after reaching a steady state lmbda
        self.c_tone = zeros(self.n)
        self.T_b  = ones(self.n)

        # # INITIAL PARAMETERS
        self.T_cur    = 0
        self.myo_term = 0
        self.tau_term = 0
        self.met_term = 0

        # solution variables
        self.lmbda   = ones(self.n)
        self.theta   = full(self.n, 0.5)


    def steady_state_solve(self, tol=1e-12):
        '''
        At steady state, for every generation:
            - d[theta]/dt == 0
            - d[lmbda]/dt == 0
            - S_tone      == 0
            - theta       == 0.5
        Finds the lmbda (root nearest 1.0) where the wall tension balances the
        current fluid tension, then stores the reference tone and tension.
        '''

        theta_ss = 0.5

        # ? residual for all generations at once (x has shape (n,))
        residual = lambda x: self.T_cur - (
            self._Tpassive(x) + theta_ss * self._Tactive(x)
        )

        # ? evaluate on a grid of physiological lmbda: res has shape (n, G)
        grid = linspace(0.2, 5.0, 5)
        res  = asarray([residual(full(self.n, g)) for g in grid]).T

        # ? sign changes, per generation
        sc      = res[:, :-1] * res[:, 1:] <= 0
        has_root = sc.any(axis=1)
        if not all(has_root):
            bad = self.gen[~has_root]
            raise RuntimeError(f'No root for lmbda found for generation(s) {bad}.')

        # ? bracket nearest lmbda = 1 for each generation
        dist = where(sc, abs(grid[:-1] - 1.0), inf)
        i    = argmin(dist, axis=1)
        a, b = grid[i], grid[i + 1]

        # ? vectorised bisection on the brackets
        fa     = residual(a)
        n_iter = int(ceil(log2((grid[1] - grid[0]) / tol))) + 1
        for _ in range(n_iter):
            m    = 0.5 * (a + b)
            fm   = residual(m)
            left = fa * fm <= 0                 # root lies in [a, m]
            b    = where(left, m, b)
            a    = where(left, a, m)
            fa   = where(left, fa, fm)

        # ? update solution attributes
        self.lmbda = 0.5 * (a + b)
        self.theta = full(self.n, theta_ss)


    def solve(self):
        n     = self.n
        y_old = concatenate((self.lmbda, self.theta))
        sol = solve_ivp(
            lambda t, y: self._rates(y[:n], y[n:]), 
            (0, self.dt), 
            y_old,
            method='Radau', 
            rtol=1e-6, 
            atol=1e-9
        )
        y = sol.y[:, -1]
        self.lmbda, self.theta = y[:n], y[n:]


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
        dlmbda = (self.T_cur - total_tension) / (self.tau_lmbda * self.T_b)

        # return dlmbda, dtheta
        return concatenate((dlmbda, dtheta))


    def _Tpassive(self, lmbda):
        return self.cpas1 * exp( self.cpas2*( lmbda-1 ) )


    def _Tactive(self, lmbda):
        return self.cact1 * exp( 
            - ( ( lmbda - self.cact2 ) / self.cact3 )**2 
        )


    def set_basal_state(self):
        '''
        Set basal tone and tension values for a current steady state.
        '''

        # ? basal tone and tension 
        self.c_tone = self.tau_term + self.met_term - self.myo_term
        self.T_b  = self.T_cur.copy()

        # ? basal lmbda
        # ! store basal dilated state as a reference
        # ! may used this for plotting, i.e. plot the relative dilation to rest
        # ! but lmbda will still be relative to the initial fractal geometry
        self.lmbda_b = self.lmbda.copy()


    def update_coefficients(self, r):
        '''
        r : array of vessel radii [μm], one per regulated generation.
        Call again if the reference radii change (e.g. after 
        wk.set_reference()).
        '''

        d = 2 * asarray(r, dtype=float)

        # outside = (d < 40) | (d > 180)
        # if any(outside):
        #     warnings.warn(
        #         'Coefficient fits are valid for d = 40-180 um. Generation(s) '
        #         f'{self.gen[outside]} have d = {d[outside].round(0)} μm.'
        #     )

        self._compute_tension_coefficients(d)
        self._compute_tone_coefficients(d)
    

    def _compute_tension_coefficients(self, d):
        '''
        r : vessel radius [um]. 
        Valid range d ~ 40-180 micro meters.
        Sets cpas1, cpas2, cact1, cact2, cact3 from diameter, d [μm].
        '''

        # ? fitted magnitudes
        self.cpas1 = -0.08507 + 6.666249*d  # linear, R2=1.000000
        self.cact1 = 1.30007  * d**1.47995  # power law, R2=0.999999

        # ? shape parameters
        self.cpas2 = 12.51775 - 0.026982*d      # linear, R2=0.999999
        self.cact2 = 0.77296  - 0.000590*d      # linear, R2=1.000000
        self.cact3 = 0.41499  - 0.000800*d      # linear, R2=1.000000


    def _compute_tone_coefficients(self, d):
        '''
        r : vessel radius [um]. 
        Valid range 2r = d ~ 40-180 micro meters.
        Sets cmyo and ctau from diameter.
        '''

        self.cmyo  = 1.3665 / d;
        self.ctau  = 0.0258 * d;
    

    def _as_array(self, x):
        return broadcast_to(asarray(x, dtype=float), (self.n,)).copy()


    @property
    def T_cur(self):
        return self._T_cur

    @T_cur.setter
    def T_cur(self, tension):
        self._T_cur = self._as_array(tension)


    @property
    def myo_term(self):
        return self._myo_term

    @myo_term.setter
    def myo_term(self, tension):
        self._myo_term = self.cmyo * self._as_array(tension)


    @property
    def tau_term(self):
        return self._tau_term

    @tau_term.setter
    def tau_term(self, shear):
        self._tau_term = self.ctau * self._as_array(shear)


    @property
    def met_term(self):
        return self._met_term

    @met_term.setter
    def met_term(self, oxygen):
        self._met_term = self.cmet * self._as_array(oxygen)





    # def passive_stretch(self, P):
    #     '''
    #     Passive (tone-free) stretch lmbda_p = Dp(P)/Dp100 at pressure P [Pa],
    #     from exp(cpas2 (lmbda_p - 1)) = (P / 100 mmHg) lmbda_p, per generation.
    #     Uses the coefficients evaluated at Dp100. Vectorised bisection.
    #     '''

    #     p = self._as_array(P) / P100
    #     f = lambda l: exp(self.cpas2 * (l - 1.0)) - p * l

    #     a = full(self.n, 0.02)
    #     b = full(self.n, 3.0)
    #     if any(f(a) * f(b) > 0):
    #         raise RuntimeError('No passive stretch bracket (check pressure and Dp100).')

    #     fa = f(a)
    #     for _ in range(60):
    #         m    = 0.5 * (a + b)
    #         fm   = f(m)
    #         left = fa * fm <= 0
    #         b    = where(left, m, b)
    #         a    = where(left, a, m)
    #         fa   = where(left, fa, fm)

    #     return 0.5 * (a + b)



    # def stiffness_ratio(self):
    #     '''
    #     d(T_wall)/d(lmbda) at the basal state (theta = 0.5), divided by T_cur.
    #     Radius is lmbda * r_basal, so at fixed pressure the fluid tension grows
    #     as T_cur per unit lmbda (T = p r). A ratio above 1 means the wall is
    #     stiffer than the load and the basal state is stable.
    #     '''
    #     lb    = self.lmbda_b
    #     dTpas = self.cpas1 * self.cpas2 * exp(self.cpas2 * (lb - 1.0))
    #     dTact = self._Tactive(lb) * (-2.0 * (lb - self.cact2) / self.cact3**2)
    #     return lb * (dTpas + 0.5 * dTact) / self.T_cur




    # def calibrate(self, lmbda_b=1.0):
    #     '''
    #     Scale cpas1 and cact1 of every generation by one factor,

    #         scale = T_cur / ( cpas1_fit e^{cpas2 (lmbda_b - 1)}
    #                           + 0.5 cact1_fit exp(-((lmbda_b - cact2)/cact3)^2) ),

    #     so that the wall tension at the basal stretch lmbda_b and theta = 0.5
    #     equals T_cur. The ratio cact1/cpas1 and all shape parameters are
    #     preserved. Also sets the reference tone (c_tone) and tension (T_ref).
    #     Call once at the basal state, after T_cur, myo_term and tau_term are
    #     set. lmbda_b is a scalar or an array (one value per generation); 1
    #     puts the basal state at Dp100. Safe to repeat.
    #     '''

    #     theta_ss = 0.5

    #     if any(self.T_cur <= 0):
    #         raise ValueError('T_cur must be positive in every generation to calibrate.')

    #     self.lmbda_b = self._as_array(lmbda_b)
    #     lb           = self.lmbda_b

    #     Tw_fit = (self.cpas1_fit * exp(self.cpas2 * (lb - 1.0))
    #               + theta_ss * self.cact1_fit * exp(-((lb - self.cact2) / self.cact3)**2))
    #     self.scale = self.T_cur / Tw_fit
    #     self.cpas1 = self.scale * self.cpas1_fit
    #     self.cact1 = self.scale * self.cact1_fit

    #     # ? reference tone and tension; S_tone = 0 at the basal state
    #     self.c_tone = self.tau_term + self.met_term - self.myo_term
    #     self.T_b  = self.T_cur.copy()

    #     # ? basal state (lmbda is relative to the basal radius)
    #     self.lmbda = ones(self.n)
    #     self.theta = full(self.n, theta_ss)











