"""
auth: L. J. Barratt
date: 02/09/26
"""

#%% PACKAGES
from numpy import exp


#%% CLASS
class DirectToneRegulation:

    def __init__(self, dt, con):

        # # SOLUTION PARAMETERS
        self.dt = dt
        
        # # CONTRIBUTIONS TO TONE
        self.k_myo = con.get('myogenic',  [0])[0]
        self.k_tau = con.get('shear',     [0])[0]
        self.k_met = con.get('metabolic', [0])[0]

        # ? basal tone in the reference state
        self.c_tone = 0

        # ? intital values
        self.myo_term = 0
        self.tau_term = 0
        self.met_term = 0

        # # INITIAL CONDITION
        self.lmbda = 1.0


    def steady_state_solve(self):
        '''
        At steady state, 
            - d[lmbda]/dt == 0
            - c_tone = ...
        '''

        self.c_tone = self.tau_term + self.met_term - self.myo_term 

    
    def solve(self):

        # ? compute tone
        S_tone = self.myo_term - self.tau_term - self.met_term + self.c_tone

        # # SOLVE GRADIENT OF LAMBDA
        dlmbda = S_tone

        # # UPDATE OLD SOLUTIONS
        self.lmbda += self.dt * dlmbda


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

