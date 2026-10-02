"""
auth: L. J. Barratt
date: 25/09/26
"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
from mpi4py.MPI import UNDEFINED

# * INTERNAL LIBRARIES
from lib.solvers.base_solver         import BaseSolver
from lib.solvers.arterial_network_wk import ArterialNetworkWK
from lib.solvers.microvascular_tone  import MicrovascularTone
from lib.solvers.skeletal_muscle     import SkeletalMuscle


#%% CLASS

class PADOVA(BaseSolver):

    models = [
        'Artery', 
        'Windkessel',
        'MicrovascularNetwork', 
        'ToneRegulation', 
        'OxygenTransport', 
        'CellularMetabolism'
    ]
            

    def configure_mpi(self, comm):

        super().configure_mpi(comm)

        # ? create a subcomm for the muscle model
        if self.comm is not None:
            color = 0 if self.rank == 0 else UNDEFINED
            self.mcomm = self.comm.Split(color=color, key=0)
        else:
            self.mcomm=None
    

    def create_system(self):

        # # CREATE ARTERIAL NETWORK & WINDKESSEL(S)
        # todo run on all ranks, use global comm

        self.macro = ArterialNetworkWK(
            comm=self.comm,
            io=self.io,
            tend=self.tend
        )

        # # CREATE TONE-REGULATING NETWORK(S)
        # todo multiple, run of all ranks, use global comm

        self.micro = list(range(len(self.terminals)))
        for idx in self.terminals:

            # ! add stuff to make this work here, please
            # give some indication of which io it uses

            self.micro[idx] = MicrovascularTone(
                comm=self.comm,
                io=self.io,
                tend=self.tend
            )

        # # CREATE MUSCLE MODEL
        # todo run on lead rank only, use split comm on lead

        self.muscle = SkeletalMuscle(
            comm=self.mcomm,
            io=self.io,
            tend=self.tend
        )

        # # STORE PARAMETERS        
        # * Time Parameters
        self.t    = 0                            # initial time [s]

        # ? general time step: tone==muscle
        self.dt   = self.macro.dt                # time step size [s]
        self.Nt   = round( self.tend / self.dt )+1 # number of steps


    def initialise_solver(self):
    # outer loop (vascular tone / geometry, slow):
        # fractal tree solve → R1, R2, C for each bed (fixed for inner loop)
        # inner loop (periodic hemodynamics, fast dt ~1e-4):
            # march 1D Navier–Stokes + Windkessel network over one cardiac
            # period at fixed R,C and fixed (rest) ATP demand, repeat periods
            # until the waveform stops changing cycle-to-cycle
            # accumulate Q_out(t) over the last converged period, average it
        # muscle O2/CM steady state (algebraic, dt-free):
            # feed Q_out_avg per bed into O2.steady_state_solve(...)
            # (via compute_permeability(), since PS = f(flow))
        # compute wall shear / wall tension from the converged periodic
            # arterial solution, and free O2 from the converged muscle state
        # update lambda from wall mechanics + O2 → update fractal dilation
        # check: has lambda (equivalently R1,R2,C) stopped changing?
            # no  → repeat outer loop
            # yes → done, system at full resting steady state

        raise NotImplementedError


    def time_evolution(self, pbar):
        raise NotImplementedError


    def step(self, step):

        # # DISTAL CONDITION
        # todo update muscle atp utilisation

        # # 1. UPDATE MACROVASCULATURE & WINDKESSELS
        # todo check if femoral blood flow has changed
        # todo check if peripheral states have changed
        # * IF YES:
        #      step until t_macro == t_micro
        #      t_macro will begin at the start at of the current heart cycle
        # * IF NO:
        #       skip!

        # # 2. UPDAPTE MICROVASCULATURE

        # TODO TESTING IDEA
        # have the time step of the micro models be a function of the heart rate
        # set a maximum time step. but have it vary to match the cycle period.



        raise NotImplementedError


    def compute_network_state(self, idx):

        tension, shear, flow = self.micro[idx].compute_network_state() 
        rtot = self.micro[idx].bed.network.compute_tree_resistance()
        ceff = self.micro[idx].bed.network.compute_tree_compliance()

        return tension, shear, flow, rtot, ceff


    def update_windkessel_parameters(self):
        # ! only change R2 and C
        # ! R1 stays constant, perhaps use rho*c0/A0 


        raise NotImplementedError


    def update_fractal_pressure(self):
        # todo update fractal network root pressure using the windkessel pressure?

        raise NotImplementedError


    