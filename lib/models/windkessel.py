"""
auth: L. J. Barratt
date: 02/09/26

Windkessel model intended to be coupled to a 1D Navier--Stokes blood flow model.
"""

#%% PACKAGES

from scipy.integrate import solve_ivp

#%% CLASS

class Windkessel:

    def __init__(self, dt, pin, vasc, fluid, wkid=0):

        self.windkessel_id = wkid # matches the terminal id of the artery

        # # TIME PARAMETERS
        self.dt = dt

        # # VASCULAR PROPERTIES
        self.R1 = vasc['r1']

        # ? can be time-dependent, updated by solver
        self.R2 = vasc['r2']
        self.C  = vasc['c']

        # # FLUID PROPERTIES
        self.pcap = fluid['capacitor_pressure'] # initial pressure at t=0
        self.pout = fluid['outflow_pressure']   # constant

        self.pin = pin
        
        self.qin  = 0
        self.qout = 0

        # # NUMERICAL METHOD
        # self.solve = self._forward_euler_solve
        self.solve = self._ivp_solve


    def update_root_pressure(self, pressure):
        self.pin = pressure


    def steady_state_solve(self):
        self.pcap = (
            (self.pin/self.R1 + self.pout/self.R2) / ( 
                1.0/self.R1 + 1.0/self.R2
            )
        )

        self.qin  = (self.pin  - self.pcap) / self.R1
        self.qout = (self.pcap - self.pout) / self.R2
 

    def _forward_euler_solve(self):
        '''
        Forward-Euler time scheme.
        '''

        # # FORM & SOLVE SYSTEM:
        # ? flow through first resistor
        self.qin   = ( self.pin - self.pcold ) / self.R1

        # ? flow through second resistor
        self.qout  = ( self.pcold - self.pout ) / self.R2

        # ? capictor pressure
        dpcap      = ( self.qin - self.qout ) / self.C

        # # UPDATE SOLUTION
        self.pcap += self.dt * dpcap


    def _ivp_solve(self):
        
        pin = self.pin
        R1  = self.R1
        R2  = self.R2
        C   = self.C

        # # COMPUTE SOLUTIONS
        y_old = [self.pcap]
        sol   = solve_ivp(
            lambda t, y: self._rates(*y, pin, R1, R2, C), 
            (0, self.dt), 
            y_old,
            method='Radau', 
            rtol=1e-6, 
            atol=1e-9
        )
        self.pcap = sol.y[0,-1]

        # update resistor blood flows
        self.qin  = ( pin - self.pcap ) / R1
        self.qout = ( self.pcap - self.pout ) / R2


    def _rates(self, pcap, pin, r1, r2, c):

        qin   = ( pin  - pcap      ) / r1
        qout  = ( pcap - self.pout ) / r2
        dpcap = ( qin  - qout      ) / c

        return [dpcap]


    @property
    def R2(self):
        return self._R2

    @R2.setter
    def R2(self, value):
        self._R2 = value


    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        self._C = value


