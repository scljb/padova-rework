"""
auth: L. J. Barratt
Created on 28/09/25

Class housing methods for solving cellular metabolism in skeletal muscle.
"""

#%% PACKAGES
from scipy.optimize  import root
from scipy.integrate import solve_ivp


#%% CLASS

class CellularMetabolism:

    def __init__(self, dt, met):
        
        # # SOLUTION PARAMETERS
        self.dt   = dt              # Time step [min]

        # # PHOSPHAGEN SYSTEM
        self.vckf = met['vckf'][0]  # Max forward flux of CK reaction [mM/min]
        self.vckr = met['vckr'][0]  # Max reverse flux of CK reaction [mM/min]
        self.kb   = met['kb'][0]    # [mM]
        self.kp   = met['kp'][0]    # [mM]
        self.kia  = met['kia'][0]   # [mM]
        self.kib  = met['kib'][0]   # [mM]
        self.kiq  = met['kiq'][0]   # [mM]

        self.kiqkp = self.kiq * self.kp
        self.kbkia = self.kb  * self.kia

        # # OXIDATIVE PHOSPHORYLATION
        self.beta = met['beta'][0]  # P:O2 ratio
        self.vmax = met['vmax'][0]
        self.kadp = met['kadp'][0]

        # coupling to external tissue oxygen model
        if met['ko2t'][0] is not None:
            self.ko2t = met['ko2t'][0]
            self.compute_phioxphos = self._coupled_phioxphos

        else:
            self.compute_phioxphos = self._uncoupled_phioxphos

        
        # # INITAL CONDITION
        # ? atp utilisation
        self.katpase = 0

        # ? mass balance
        self.cat = met['total_ad'][0]    # Total adenosine
        self.cct = met['total_cr'][0]    # Total creatine

        # ? adenosine
        self.atp = met['tissue_atp'][0]
        self.adp = self.cat - self.atp
            
        # ? creatine
        self.pcr = met['tissue_pcr'][0]
        self.cr  = self.cct  - self.pcr

        # # NUMERICAL METHOD
        # self.solve = self._forward_euler_solve
        self.solve = self._ivp_solve


    def compute_steady_state_phioxphos(self):
        '''
        At steady state (ATP homeostasis), 
            - d<ATP>/dt   == 0
            - Phi[CK]f,r  == 0
            - Phi[ATPase] == Phi[OxPhos]
        '''

        return ( self.katpase * self.atp ) / self.beta


    def update_katpase(self, katpase):
        self.katpase = katpase


    def update_phioxphos(self, phioxphos):
        self.phioxphos = phioxphos


    def _forward_euler_solve(self):

        # # FLUX COMPUTATION
        phiatpase = self.katpase * self.atp
        phick_denom = (
            1.0 + 
            self.adp/self.kia + 
            self.atp/self.kiq + 
            self.pcr/self.kib + 
            self.adp*self.pcr/self.kbkia + 
            self.cr*self.atp/self.kiqkp
        )
        phickr = (
            ( self.vckr * self.cr  * self.atp / self.kiqkp ) / phick_denom
        )
        phickf = (
            ( self.vckf * self.adp * self.pcr / self.kbkia ) / phick_denom
        )

        # # COMPUTE GRADIENTS
        dcatp = -phiatpase + self.beta*self.phioxphos - phickr + phickf
        dcpcr = phickr - phickf

        # # COMPUTE SOLUTIONS
        self.atp += self.dt * dcatp
        self.pcr += self.dt * dcpcr

        # # MASS BALANCE
        self.adp = self.cat - self.atp
        self.cr  = self.cct - self.pcr


    def _ivp_solve(self):

        # # COMPUTE SOLUTIONS
        y_old = [self.atp, self.pcr]
        sol = solve_ivp(
            lambda t, y: self._rates(*y, self.phioxphos), 
            (0, self.dt), 
            y_old,
            method='Radau', 
            rtol=1e-6, 
            atol=1e-9
        )
        self.atp, self.pcr = sol.y[:, -1]

        # # MASS BALANCE
        self.adp = self.cat - self.atp
        self.cr  = self.cct - self.pcr


    def _rates(self, atp, pcr, phioxphos):
        adp = self.cat - atp
        cr  = self.cct - pcr

        phiatpase = self._phiatpase(atp)

        phick_denom = (
            1.0 + adp/self.kia + atp/self.kiq + pcr/self.kib +
            adp*pcr/self.kbkia + cr*atp/self.kiqkp
        )
        phickr = (self.vckr * cr  * atp / self.kiqkp) / phick_denom
        phickf = (self.vckf * adp * pcr / self.kbkia) / phick_denom

        dcatp = -phiatpase + self.beta*phioxphos - phickr + phickf
        dcpcr = phickr - phickf

        return dcatp, dcpcr


    def _phiatpase(self, atp):
        return self.katpase*atp

    
    def _coupled_phioxphos(self, adp, o2tis):
        return self.vmax * ( adp / ( self.kadp + adp ) ) * ( 
                o2tis / ( self.ko2t + o2tis ) 
        )

    
    def _uncoupled_phioxphos(self, adp):
        return self.vmax * ( adp / ( self.kadp + adp ) )


    def _phickr(self, atp, pcr, adp, cr):
        return (self.vckr * cr  * atp / self.kiqkp) / self._phick_denom


    def _phickr(self, atp, pcr, adp, cr):
        return (self.vckf * adp * pcr / self.kbkia) / self._phick_denom


    def _phick_denom(self, atp, pcr, adp, cr):
        return ( 1.0 + adp/self.kia + atp/self.kiq + pcr/self.kib +
            adp*pcr/self.kbkia + cr*atp/self.kiqkp
        )


# ! ############################################################################
# !                             DEPRECATED FUNCTIONS
# ! ############################################################################


    def __root_solve(self):

        # # COMPUTE SOLUTIONS
        y_old = (self.atp, self.pcr)
        sol = root(
            self._residual,
            y_old, 
            args=(y_old, self.dt), 
            method='hybr', 
            tol=1e-8
        )
        self.atp, self.pcr = sol.x

        # # MASS BALANCE
        self.adp = self.cat - self.atp
        self.cr  = self.cct - self.pcr


    def __residual(self, y, y_old, dt):
        atp, pcr = y
        atp_old, pcr_old = y_old

        dcatp, dcpcr = self._rates(atp, pcr)

        # backward Euler: y_{n+1} - y_n - dt*f(y_{n+1}) = 0
        return [atp - atp_old - dt*dcatp,
                pcr - pcr_old - dt*dcpcr]