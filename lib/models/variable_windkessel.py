"""
auth: L. J. Barratt
date: 30/09/26

Model of peripheral circulation using a Windkessel with variable far peripheral resistance.

Rn  (near resistance) == constant = rho*c0/A0  
C   (compliance)      == constant (estimated from fractal)
Rm  (mid resistance)  == constant (estimated from fractal)
Rf  (far resitance)   == variable (from scalable fractal)

RT(t) = Rm + Rf(t)
"""

#%% PACKAGES
from numpy import sqrt, log, pi, ones, exp
from numpy import arange, asarray, sum, cumsum, any, broadcast_to, concatenate


#%% CLASS

class VariableWindkessel:
    
    def __init__(self, dt, r0, Rn, c0, fluid, fractal, wkid=0):

        self.windkessel_id = wkid # matches the terminal id of the artery

        # # TIME PARAMETERS
        self.dt = dt

        # # FLUID
        # ? boundary pressures
        self.pin  = 0  # initialise to 0, set later
        self.pout = fluid['outflow_pressure']

        # ? fluid properties
        self.mu   = fluid['mu']     # fluid viscosity
        self.rho  = fluid['rho']    # fluid density
        self.c0   = c0              # wave speed of terminal artery

        # ? initial solution
        self.pcap = fluid['capacitor_pressure'] # initial pressure at t=0
        self.qin  = 0.0
        self.qout = 0.0

        # # FRACTAL GENERATION
        self.r0   = r0                          # root radius, r0=R_term
        lrr0      = fractal.get('lrr0')         # root length-radius ratio
        self.l0   = lrr0 * self.r0              # root length
        self.Rn   = Rn                          # near resistance, Rn=rho*C0/A0

        # ? conditions for mid-peripheral branch termination 
        # (only one of the following is required)
        self.Nmid = fractal.get('nmid', None)
        self.rmid = fractal.get('rmid', None)

        # ? conditions for far-peripheral branch termination
        # (only one of the following is required)
        self.N    = fractal.get('n', None)       # number of generations
        self.rcut = fractal.get('rcut', None)    # cut-off radius

        # ? scaling factors (optional)
        self.psi = fractal.get('psi', sqrt(0.6)) # radius scaling factor
        self.phi = fractal.get('phi', None)      # length scaling factor

        # ? create the fractal; compute Rn, C, Rm, and Rf
        self._build_fractal()


    def update_root_pressure(self, pressure):
        self.pin = pressure


    def steady_state_solve(self):
        '''
        At steady state:
            - dp/dt = qin-qout == 0
        '''

        g1, g2 = 1.0 / self.Rn, 1.0 / self.RT
        self.pcap = (self.pin * g1 + self.pout * g2) / (g1 + g2)
        self._update_flows()


    def solve(self):
        '''
        C dp/dt = (pin - p)/R1 - (p - pout)/R2 is linear with constant
        coefficients over a step, so p relaxes exponentially to p_inf.
        '''

        g1, g2 = 1.0 / self.Rn, 1.0 / self.RT
        p_inf  = (self.pin * g1 + self.pout * g2) / (g1 + g2)
        tau    = self.C / (g1 + g2)
        self.pcap = p_inf + (self.pcap - p_inf) * exp(-self.dt / tau)
        self._update_flows()


    def apply_dilation(self, lmbda):
        '''
        Scale radii of Rf compartment by lmbda. Set new RT=Rn+Rf.
        '''

        lmbda = broadcast_to(asarray(lmbda, dtype=float), self.lmbda.shape)
        if any(lmbda <= 0):
            raise ValueError('lmbda must be positive.')

        self.lmbda = lmbda.copy()
        self.Rf    = sum(self.R_far0 / self.lmbda**4)


    def set_reference(self):
        '''
        Make the current dilated state the new baseline (lmbda -> 1).
        The far radii and resistances absorb the current lmbda, so Rf, RT and
        every pressure are unchanged. Later lmbda values are applied to this new reference state.
        '''

        k = self.Nmid
        self.radius0[k:] = self.radius0[k:] * self.lmbda
        self.R_gen0[k:]  = self.R_gen0[k:] / self.lmbda**4
        self.R_far0      = self.R_gen0[k:]
        self.lmbda       = ones(self.N - k)


    def generation_state(self):
        '''
        Mean state of every generation n = 1..N at the current operating point.
        qout flows through the whole fractal and splits equally over the 2^n
        vessels of generation n.
        '''

        k     = self.Nmid
        s_all = concatenate((ones(k), self.lmbda))
        R_gen = self.R_gen0 / s_all**4      # R_gen0 already holds the baseline
        drop  = self.qout * R_gen           # pressure drop across generation n

        p_out = self.pout + (sum(drop) - cumsum(drop))
        p_in  = p_out + drop
        p_mid = 0.5 * (p_in + p_out)

        Q = self.qout / 2.0**self.gen       # flow per vessel
        r = s_all * self.radius0            # current radius
        t = 4.0 * self.mu * Q / (pi * r**3) # wall shear stress
        T = p_mid * r                       # Laplace, per unit length

        return dict(
            gen     = self.gen, # [-]
            radius  = r,        # [m]
            Q       = Q,        # [m3/s]
            p_in    = p_in,     # [Pa]
            p_out   = p_out,    # [Pa]
            p       = p_mid,    # [Pa]
            tau     = t,        # [Pa]
            tension = T         # [Pa m]
        )


    def _update_flows(self):
        self.qin  = (self.pin  - self.pcap) / self.Rn
        self.qout = (self.pcap - self.pout) / self.RT


    def _build_fractal(self):
        '''
        Symmetric bifurcating tree (Alastruey 2006, Sec. 2.8). Generation n has
        2^n identical vessels with R_n = psi^n r0 and l_n = phi^n l0.

        Generation 0 is the 1D terminal artery and is not part of the fractal.
        '''

        r0, l0, psi = self.r0, self.l0, self.psi

        # # NUMBER OF GENERATIONS (far termination)
        if self.N is None:
            if self.rcut is None:
                raise ValueError("Provide fractal['n'] or fractal['rcut'].")

            self.N = int(round(log(self.rcut / r0) / log(psi)))


        N = self.N

        # # LENGTH SCALING
        if self.phi is None:
            self.phi = (N / (N + 1.0)) / (2.0 * psi**2) # eqn:2.183

        phi = self.phi

        # # GEOMETRY PER GENERATION
        gen = arange(1, N+1)
        self.gen = gen
        self.radius0 = psi**gen * r0 # baseline radius
        self.length0 = phi**gen * l0 # baseline length

        # # RESISTANCE PER GENERATION
        Rmu0 = 8.0 * self.mu * l0 / (pi * r0**4)
        self.R_gen0 = Rmu0 * (phi / (2.0 * psi**4))**gen    # eqn:2.166-2.168

        # # TOTAL COMPLIANCE
        C0     = pi * r0**2 * l0 / (self.rho * self.c0**2)
        self.C = sum(C0 * (2.0 * phi * psi**3)**gen)        # eqn:2.177

        # # MID / FAR PERIPHERY SPLIT
        if self.Nmid is None:
            if self.rmid is None:
                raise ValueError("Provide fractal['nmid'] or fractal['rmid'].")

            self.Nmid = int(sum(self.radius0 >= self.rmid))


        k = self.Nmid
        if not 0 <= k < N:
            raise ValueError(f'nmid must satisfy 0 <= nmid < n (got {k}, {N}).')


        self.Rm     = sum(self.R_gen0[:k])   # mid resistance, fixed
        self.R_far0 = self.R_gen0[k:]        # far generations, baseline
        self.lmbda  = ones(N - k)            # current radius scaling
        self.Rf     = sum(self.R_far0)       # far resistance


    @property
    def RT(self):
        return self.Rm + self.Rf





