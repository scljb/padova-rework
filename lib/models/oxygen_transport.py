"""
auth: L. J. Barratt
Created on 12/02/26

Class housing methods for solving lumped oxygen transport in skeletal muscle.
"""

#%% PACKAGES

# * EXTERNAL PACKAGES
from numpy           import exp
from scipy.optimize  import brentq
from scipy.integrate import solve_ivp


#%% CLASS

class OxygenTransport:

    def __init__(self, dt, subject, blood):
        
        # # SOLUTION PARAMETERS
        self.dt = dt
        
        # # SUBJECT PARAMETERS
        self.vcap = subject['blood_volume'][0]
        self.vtis = subject['tissue_volume'][0]

        # # BLOOD PARAMETERS
        # ? perfusion
        self.qm_rest   = blood['flow_rest'][0]
        self.psap_rest = blood['perm_rest'][0]
        self.psap_exer = blood['perm_exer'][0]
        self.psap_tau  = blood['perm_tau'][0]

        # ? constants
        hct      = blood['hct'][0] / 100
        wmc      = blood['wmc'][0] / 100

        crbc     = blood['crbc'][0]
        cmc      = blood['cmc'][0]
        self.khb = blood['khb'][0]
        self.kmb = blood['kmb'][0]
        self.n   = blood['n'][0]

        # ? grouped coefficients for later 
        self.gammab_sub = 4 * hct * crbc * self.khb * self.n
        self.gammat_sub = wmc * cmc * self.kmb
        self.ftb_sub    = 4 * hct * crbc * self.khb

        # # INITIAL CONDITIONS
        # ? flow in
        self.qm = 0

        # ? arterial oxygen
        o2art  = blood['artery_o2'][0]
        o2artb = ( self.ftb_sub * o2art**self.n / ( 
                1 + ( self.khb * o2art**self.n ) 
            ) 
        )
        self.o2artt = o2art + o2artb

        # ? solution parameters
        self.o2cap = blood['capillary_o2'][0]
        self.o2tis = blood['tissue_o2'][0]

        # # NUMERICAL METHOD
        # self.solve = self._forward_euler_solve
        self.solve = self._ivp_solve


    def steady_state_solve(self):

        # ? steady state perfusion
        perfusion_ss = self.phioxphos * self.vtis

        # ? residual function of free o2cap
        def residual(o2cap):
            return self.qm * ( self.o2artt-self._o2capt(o2cap) ) - perfusion_ss


        o2cap_ss = brentq(residual, 1e-9, self.o2artt)

        # ? compute tissue oxygen
        o2tis_ss = o2cap_ss - perfusion_ss / self._psap()

        # ? update solution attributes
        self.o2cap = o2cap_ss
        self.o2tis = o2tis_ss


    def update_flow(self, q):
        self.qm = q


    def update_phioxphos(self, phioxphos):
        self.phioxphos = phioxphos


    def update_blood_volume(self, vol):
        self.vcap = vol


    def _forward_euler_solve(self):

        # # PARAMETER COMPUTATION
        # ? permeability from flow
        ps = self._psap()
        perfusion = ps*( self.o2cap - self.o2tis )
       
        # ? total oxygen from free & bound oxygen
        o2capt = self._o2capt(self.o2cap)

        # ? oxygen chemical balance terms
        gamma_cap = self._gammab(self.o2cap)
        gamma_tis = self._gammat(self.o2tis)

        # # COMPUTE GRADIENTS
        do2cap = (
            ( self.qm*( self.o2artt - o2capt ) - perfusion ) / ( 
                gamma_cap*self.vcap 
            )
        )

        do2tis = (
            ( perfusion - ( self.phioxphos*self.vtis ) ) / ( 
                gamma_tis*self.vtis 
            )
        )

        # # COMPUTE SOLUTIONS
        self.o2cap += self.dt * do2cap
        self.o2tis += self.dt * do2tis


    def _ivp_solve(self):

        # # COMPUTE SOLUTIONS
        y_old = [self.o2cap, self.o2tis]
        sol = solve_ivp(
            lambda t, y: self._rates(*y, self.phioxphos), 
            (0, self.dt), 
            y_old,
            method='Radau', 
            rtol=1e-6, 
            atol=1e-9
        )
        self.o2cap, self.o2tis = sol.y[:, -1]


    def _rates(self, o2cap, o2tis, phioxphos):
        
        # ? perfusion from flow + o2 gradient
        perfusion = self._psap() * (o2cap - o2tis)

        # ? oxygen utilisation from phioxphos
        uo2m = phioxphos * self.vtis

        # ? form ODEs
        do2cap = ( self.qm*( self.o2artt-self._o2capt(o2cap) ) - perfusion ) / (
                self._gammab(o2cap)*self.vcap
        )
        do2tis = ( perfusion - uo2m ) / ( self._gammat(o2tis)*self.vtis )

        return do2cap, do2tis


    def _o2capt(self, o2f):

        o2b = self.ftb_sub * o2f**self.n / ( 
            1 + ( self.khb * o2f**self.n ) 
        ) 
        
        return o2f + o2b
    

    def _gammab(self, o2f):
        return 1 + ( 
            ( self.gammab_sub * o2f**( self.n - 1 ) ) / ( 
                1 + self.khb * o2f**self.n 
            )**2 
        )


    def _gammat(self, o2f):
        return 1 + ( ( self.gammat_sub ) / ( 1 + self.kmb * o2f )**2 )


    def _psap(self):
        return (
            self.psap_rest + ( self.psap_exer - self.psap_rest ) * ( 
                1.0 - exp( -(self.qm-self.qm_rest)/self.psap_tau ) 
            )
        )