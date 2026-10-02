"""
auth: L. J. Barratt
date: 02/09/26
"""

#%% PACKAGES
from numpy import exp


#%% CLASS
class RelativeToneRegulation:

    def __init__(self, dt, con, comm=None):

        # # SERIAL ONLY
        self.comm = comm
        if ( comm is not None ) and ( comm.Get_size() > 1 ):
            raise RuntimeError('too many ranks')

        # # SOLUTION PARAMETERS
        self.dt = dt

        # # CONTRIBUTIONS TO TONE
        self.k_myo = con.get('myogenic',  [0])[0]
        self.k_tau = con.get('shear',     [0])[0]
        self.k_met = con.get('metabolic', [0])[0]
        self.k_rec = con.get('recoil',    [0])[0]

        # ? basal tone in the reference state
        self.c_tone = 0

        # ? reference values
        self.tension_ref = 0.1
        self.shear_ref   = 0.1
        self.oxygen_ref  = 0.1

        # ? intital values
        self.tension_current = 0
        self.shear_current   = 0
        self.oxygen_current  = 0

        # # INITIAL CONDITION
        self.lmbda = 1.0


    def _myo_contribution(self):
        return self.tension_current / ( self.tension_ref * self.lmbda**3 ) - 1


    def _tau_contribution(self):
        return max(
            self.shear_current / ( self.shear_ref * self.lmbda**3 ) - 1,
            0           # low shear will not cause constriction
        )


    def _met_contribution(self):
        return max(
            (self.oxygen_ref - self.oxygen_current) / self.oxygen_ref,
            0           # high oxygen will not cause contriction
        )

    
    def solve(self):

        # # SOLVE GRADIENT OF LAMBDA
        dlmbda = ( 
            - self.k_myo * self._myo_contribution()
            + self.k_tau * self._tau_contribution()
            + self.k_met * self._met_contribution() 
            - self.k_rec * ( self.lmbda - 1 )
        )

        # # UPDATE OLD SOLUTIONS
        self.lmbda += self.dt * dlmbda


    def update_reference_tension(self, tension):
        self.tension_ref = tension
    

    def update_current_tension(self, tension):
        self.tension_current = tension


    def update_reference_shear(self, tau):
        self.shear_ref = tau
    

    def update_current_shear(self, tau):
        self.shear_current = tau


    def update_reference_oxygen(self, o2):
        self.oxygen_ref = o2


    def update_current_oxygen(self, o2):
        self.oxygen_current = o2

