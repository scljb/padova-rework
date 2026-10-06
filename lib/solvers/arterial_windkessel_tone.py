"""
auth: L. J. Barratt
date: 02/09/26
"""

#%% PACKAGES

#* EXTERNAL LIBRARIES
from scipy.optimize import newton
from numpy          import pow
from mpi4py.MPI     import COMM_SELF
import sys

#* INTERNAL LIBRARIES
from lib.solvers.arterial_network import ArterialNetwork
from lib.solvers.windkessel_tone  import WindkesselTone


#%% CLASS

class ArterialWindkesselTone(ArterialNetwork):

    models = [
        'Artery', 
        'VariableWindkessel'
        'VectorisedToneRegulation'
    ]

    def create_system(self):

        # # INITIALISE ARTERIAL NETWORK
        super().create_system()

        # # INITIALISE WINDKESSEL+TONE MODEL(s)
        wk_io = self.io.model_io['VariableWindkessel']
        self.wk_models = {}
        
        # instance windkessel class for each terminal
        for ii in self.local_terminals:
            artery = self.arteries[ii]

            # ? check terminal outflow model is a windkessel
            if self.arteries[ii].outflow_model != 'wk':
                raise NotImplementedError('mismatched boundary types are not supported')

            # ? require equal time steps. if none present, set equal
            wkdt = wk_io.solution.get('dt', [None])[0]
            if ( wkdt is not None ) and ( wkdt != self.dt ):
                raise ValueError('time steps do not match')

            # ? get properties of the terminal feed artery
            r0     = artery.vessel['rmin']
            c0     = artery.c0o
            Rn     = artery.rho * c0 / artery.A0o

            # ? instance class
            self.wk_models[ii] = WindkesselTone(
                comm=COMM_SELF,
                io=self.io,
                tend=self.tend,
                extras=(self.dt, r0, c0, Rn, ii)
            )


    def initialise_solver(self, called=False):

        # # INITIALISE FLOW IN THE LARGE VESSELS
        super().initialise_solver()

        # # INITIALISE FLOW IN THE SMALL VESSELS
        for ii, wkt in self.wk_models.items():
            artery = self.arteries[ii]
            Pmean  = 0

            wkt.initialise_solver(called=True, pressure=Pmean)

        
        # ? write initial solution
        if not called:
            self.m_attrs = self.get_model_dict(a=self)
            self.io.write_solutions(-1, 0, **self.m_attrs)


    def time_evolution(self, pbar):

        for step in pbar:

            pbar.set_postfix(t=self._postfix())

            # # SOLVE ARTERIAL NETWORK
            # ? update root & junction condition
            self.update_root_condition(step)
            self.update_junction_condition()

            # ? update network solution
            self.step(step, write=True)

            # # SOLVE TERMINAL SUB-SYSTEMS
            self.update_terminal_condition(step)


    def update_terminal_condition(self, step):

        for ii in self.local_terminals:

            # # SOLVE EXPLICIT COUPLING TO RESISTIVE SEGMENT
            artery = self.arteries[ii]
            wk     = self.wk_models[ii]

            # ? solutions at outlet
            Al, Ul = artery.Un(artery.L,)

            # ? reference values at outlet
            A0   = artery.A0(artery.L,)
            rho  = artery.rho
            beta = artery.beta0(artery.L,)
            pd   = artery.pd

            # ? peripheral properties
            r1 = wk.Rn
            pc = wk.pcap

            # ? compute new boundary solution
            As = newton(
                func=self._coupling_residual,
                x0=Al,
                tol=1e-10,
                maxiter=100,
                args=(Al,Ul,A0,rho,beta,r1,pd,pc)
            )

            # ? calculate area of 'ghost node'
            Ar = pow(2*pow(As,0.25) - pow(Al,0.25),4)

            # ? apply condition (velocity is assumed zero-gradient)
            artery.outlet.assign(Ar)

            # # SOLVE WINDKESSEL + TONE MODEL (USING BOUNDARY AREA SOLUTION)
            P_As = pd + ( beta/A0 ) * ( pow(As,0.5) - pow(A0,0.5) )
            wk.step(step, P_As)


    @staticmethod
    def _coupling_residual(As, Al, Ul, A0, rho, beta, r1, pd, pc):
    
        # # PRESSURE AT BOUNDARY VIA TUBE LAW
        Pe  = pd + ( beta/A0 ) * ( pow(As,0.5) - pow(A0,0.5) )

        # # FORM RESIDUAL FUNCTION
        c = pow(beta/2/rho/A0,0.5)
        cAl = c * pow(Al,0.25)  # Wave speed inside the 1D domain
        cAs = c * pow(As,0.25)  # Wave speed at the boundary
        
        F_1 = r1 * ( Ul + 4 * cAl ) * As
        F_2 = r1 * ( 4 * cAs * As )

        return F_1 - F_2 - Pe + pc




