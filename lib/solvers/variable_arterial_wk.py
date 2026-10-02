"""
auth: L. J. Barratt
date: 02/09/26
"""

#%% PACKAGES

#* EXTERNAL LIBRARIES
from scipy.optimize                 import newton
from numpy                          import pow
import sys

#* INTERNAL LIBRARIES
from lib.solvers.arterial_network   import ArterialNetwork
from lib.models.variable_windkessel import VariableWindkessel


#%% CLASS

class VariableArterialWK(ArterialNetwork):

    models = ['Artery', 'VariableWindkessel']

    def create_system(self):

        # # INITIALISE ARTERIAL NETWORK
        super().create_system()

        # # INITIALISE WINDKESSEL MODEL(S)
        wk_io = self.io.model_io['VariableWindkessel']
        
        # instance windkessel class for each terminal
        self.wk_models = list(range(len(self.local_terminals)))
        for ii in self.local_terminals:

            # ? check terminal outflow model is a windkessel
            if self.arteries[ii].outflow_model != 'wk':
                raise NotImplementedError('mismatched boundary types are not supported')

            # ? require equal time steps. if none present, set equal
            wkdt = wk_io.solution.get('dt', [None])[0]
            if ( wkdt is not None ) and ( wkdt != self.dt ):
                raise ValueError('time steps do not match')

            # ? get the inputs for this model
            fluid = self.extract_values(wk_io.fluid,   0)
            frctl = self.extract_values(wk_io.fractal, 0)

            # ? get properties of the terminal feed artery
            artery = self.arteries[ii]
            r0     = artery.vessel['rmin']
            c0     = artery.c0o
            Rn     = artery.rho * c0 / artery.A0o

            # ? instance class
            self.wk_models[ii] = VariableWindkessel(
                self.dt, 
                0, r0, Rn, c0,
                fluid, frctl,
                wkid=ii
            )


    def time_evolution(self, pbar):

        for step in pbar:

            pbar.set_postfix(t=self._postfix())

            # # SOLVE COUPLED SYSTEM
            self.step(step, write=True)
        

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

            # # UPDATE WINDKESSEL MODEL (USING BOUNDARY AREA SOLUTION)
            P_As = pd + ( beta/A0 ) * ( pow(As,0.5) - pow(A0,0.5) )
            wk.update_root_pressure(P_As)
            wk.solve()

            # ? write windkessel solution to file
            self.io.model_io['VariableWindkessel'].write_solution(
                step, self.t, wk
            )


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




