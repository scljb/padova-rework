"""
auth: L. J. Barratt
date: 25/09/26
"""

#%% PACKAGES
from numpy import exp, pi, sin
from numpy import arange, sum, asarray, atleast_1d

#%% CLASS

class TimeFunction:

    _FORMS = {
        'exponential', 
        'gaussian', 
        'sinusoidal', 
        'step', 
        'box',
        'constant'
    }

    def __init__(self, params):

        self.form = params['form'][0]
        if self.form not in self._FORMS:
            raise ValueError(f"'{self.form}' is not a recognized form. "
                              f"Available: {sorted(self._FORMS)}")


        # # SETUP TIME-DEPENDENT FUNCTION
        setup = getattr(self, f'_setup_{self.form}')
        setup(params)

        # ? set updater
        self._updater = getattr(self, f'_eval_{self.form}')


    def compute_value(self, t):
        return self._updater(t)

    
# # ######################### #################### #########################
# # ######################### EXPONENTIAL FUNCTION #########################
# # ######################### #################### #########################


    def _setup_exponential(self, params):
        self.y1    = params['initial'][0]
        self.dy    = atleast_1d(asarray(params['change'], dtype=float))
        self.delta = atleast_1d(asarray(params['delta'], dtype=float))
        self.tau   = atleast_1d(asarray(params['tau'], dtype=float))

        if not (self.dy.shape == self.delta.shape == self.tau.shape):
            raise ValueError('Given lists of non-equal lengths.')
        

    def _eval_exponential(self, t):
        alpha = t >= self.delta

        return self.y1 + sum(
            alpha * self.dy * (1.0 - exp(-(t - self.delta) / self.tau))
        )


# # ########################## ################## ##########################
# # ########################## GUASSIAN  FUNCTION ##########################
# # ########################## ################## ##########################


    def _setup_gaussian(self, params):
        self.amp   = params['amplitude'][0]
        self.sigma = params['sigma'][0]
        ncycles    = params['ncycles'][0]
        HR         = params['hr'][0]

        # frequency of wave
        freq       = HR / 60

        # gaussian centres
        self.c     = 3 * self.sigma + arange(ncycles) / freq


    def _eval_gaussian(self, t):
        return self.amp * sum(
            exp(-((t - self.c) / self.sigma) ** 2)
        )


# # ######################### #################### #########################
# # ######################### SINUSOIDAL  FUNCTION #########################
# # ######################### #################### #########################


    def _setup_sinusoidal(self, params):
        self.mean = params['mean'][0]
        self.amp  = params['amp'][0]
        HR        = params['hr'][0]

        # frequency of wave
        self.freq = HR / 60.0


    def _eval_sinusoidal(self, t):
        return self.mean + self.amp * sin(2 * pi * self.freq * t)


# # ############################ ############## ############################
# # ############################ STEP  FUNCTION ############################
# # ############################ ############## ############################


    def _setup_step(self, params):
        self.y1    = params['value1'][0]
        self.y2    = params['value2'][0]
        self.delta = params['delta'][0]


    def _eval_step(self, t):
        alpha = t >= self.delta

        return self.y1 + alpha*( self.y2-self.y1 )


# # ############################# ############ #############################
# # ############################# BOX FUNCTION #############################
# # ############################# ############ #############################


    def _setup_box(self, params):
        self.y1     = params['y1'][0]
        self.dy     = atleast_1d(asarray(params['dy'], dtype=float))
        self.delta1 = atleast_1d(asarray(params['delta1'], dtype=float))
        self.delta2 = atleast_1d(asarray(params['delta2'], dtype=float))


    def _eval_box(self, t):
        active = (t >= self.delta1) & (t < self.delta2)

        return self.y1 + (active * self.dy).sum()

    
# # ############################### ######## ###############################
# # ############################### CONSTANT ###############################
# # ############################### ######## ###############################


    def _setup_constant(self, params):
        self.y = params['value'][0]


    def _eval_constant(self, t):
        return self.y

