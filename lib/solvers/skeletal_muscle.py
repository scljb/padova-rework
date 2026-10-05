"""
auth: L. J. Barratt
date: 25/02/26

Class housing methods for solving lumped oxygen transport in skeletal muscle.
"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
from numpy  import exp
from scipy.integrate import solve_ivp
import sys

# * INTERNAL LIBRARIES
from lib.solvers.base_solver        import BaseSolver

from lib.models.oxygen_transport    import OxygenTransport
from lib.models.cellular_metabolism import CellularMetabolism

from lib.etc.time_function          import TimeFunction


#%% CLASS

class SkeletalMuscle(BaseSolver):

    models = ['OxygenTransport', 'CellularMetabolism']

    def configure_mpi(self, comm):

        super().configure_mpi(comm)

        if ( comm is not None ) and ( comm.Get_size() != 1 ):
            raise TypeError('SkeletalMuscle does not support MPI.')
            

    def create_system(self):

        oxy_io = self.io.models['OxygenTransport']
        met_io = self.io.models['CellularMetabolism']

        # # STORE PARAMETERS
        # * Time parameters
        self.t = 0

        # ? require equal time steps
        if oxy_io.solution['dt'][0] == met_io.solution['dt'][0]:
            self.dt   = oxy_io.solution['dt'][0]
        else:
            raise ValueError('time steps in muscle models do not match')


        self.Nt   = round( self.tend / self.dt )+1  # No. of time steps

        # ? convert time units, this solver uses minute units for rates
        self.dtmin   = self.dt   / 60

        # # RATE OF ATP UTILISATION
        self.katpase = TimeFunction(met_io.atpase)

        # # CONFIGURE OXYGEN TRANSPORT
        self.O2 = OxygenTransport(
            self.dtmin,
            oxy_io.subject,
            oxy_io.blood
        )

        # # CONFIGURE CELLULAR METABOLISM
        self.CM = CellularMetabolism(
            self.dtmin,
            met_io.metabolism
        )

        # # NUMERICAL METHOD
        self.step = self.step
        # self.step = self.coupled_step


    def initialise_solver(self, called=False):

        # # SET INITIAL CONDITIONS
        if not called:
            self.update_muscular_flow()
            self.update_atp_utilisation()
        
        # # METABOLIC SYSTEM IN ATP HOMEOSTASIS
        oxphos = self.CM.compute_steady_state_phioxphos()
        self.O2.update_phioxphos(oxphos)
        self.CM.update_phioxphos(oxphos)

        # # COMPUTE STEADY STATE MUSCLE OXYGENATION
        self.O2.steady_state_solve()

        # ? write initial condition
        if not called:
            self.m_attrs = self.get_model_dict(a=self)
            self.io.write_solutions(-1, 0, **self.m_attrs)


    def step(self, step):
        '''
        Semi-implicit coupling.
        '''

        # ? time varying parameters:
        self.update_muscular_flow()
        self.update_atp_utilisation()

        # # SOLVE OXYGEN TRANSPORT SYSTEM
        self.O2.solve()

        # # MODEL COUPLING
        self.update_oxygen_utilisation()

        # # SOLVE METABOLISM SYSTEM
        self.CM.solve()

        # # UPDATE TIME
        self.t += self.dt


    def update_muscular_flow(self):
        flow = self.rc.compute_value(self.t)
        self.O2.update_flow(flow)


    def update_atp_utilisation(self):
        katpase = self.katpase.compute_value(self.t)
        self.CM.update_katpase(katpase)


    def update_oxygen_utilisation(self):
        oxphos = self.CM.compute_phioxphos(self.CM.adp, self.O2.o2tis)
        self.O2.update_phioxphos(oxphos)
        self.CM.update_phioxphos(oxphos)


    def coupled_step(self, step):
        '''
        Implicit coupling.
        '''

        # # SOLVE COUPLED SYSTEM
        y0 = [self.CM.atp, self.CM.pcr, self.O2.o2cap, self.O2.o2tis]

        sol = solve_ivp(
            self._combined_rates,
            (0, self.dt),
            y0,
            method='Radau',
            rtol=1e-8,
            atol=1e-12,
            # dense_output=True
        )
        atp, pcr, o2cap, o2tis = sol.y[:, -1]

        if not sol.success:
            raise RuntimeError(f'Coupled muscle solve failed: {sol.message}')


        # ? set solution variables
        self.CM.atp, self.CM.pcr = atp, pcr
        self.CM.adp = self.CM.cat - atp
        self.CM.cr  = self.CM.cct - pcr

        self.O2.o2cap, self.O2.o2tis = o2cap, o2tis

        # ? update phioxphos for both objects
        self.update_oxygen_utilisation()

        # # UPDATE TIME
        self.t += self.dt

    def _combined_rates(self, t, y):
        atp, pcr, o2cap, o2tis = y
        adp = self.CM.cat - atp

        # ? coupling term: OxPhos flux
        phioxphos = self.CM.compute_coupled_phioxphos(adp, o2tis)

        # ? CM rates (pure function of atp, pcr, phioxphos)
        dcatp, dcpcr = self.CM._rates(atp, pcr, phioxphos)

        # ? O2 rates (pure function of o2cap, o2tis, phioxphos)
        do2cap, do2tis = self.O2._rates(o2cap, o2tis, phioxphos)

        return [dcatp, dcpcr, do2cap, do2tis]
