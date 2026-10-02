"""
auth: L. J. Barratt
date: 15/09/26


"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
from tqdm   import tqdm


# * INTERNAL LIBRARIES
from lib.solvers.network_tone import NetworkAutoregulation
from lib.solvers.direct_network_tone import \
        NetworkAutoregulation
from lib.solvers.skeletal_muscle        import SkeletalMuscle

from lib.solvers.base_solver import BaseSolver

#%% CLASS

class MuscleTone(BaseSolver):

    models = [
        'MicrovascularNetwork', 
        'ToneRegulation', 
        'OxygenTransport', 
        'CellularMetabolism'
    ]

    def configure_mpi(self, comm):
        if comm is not None:
            self.ncomm = comm
            self.rank  = comm.Get_rank()

            if self.rank == 0:
                self.mcomm = comm.Split(color=self.mpi.rank, key=0)
            else:
                self.mcomm = None
            
        else:
            self.rank  = 0
            self.ncomm = None
            self.mcomm = None


    def create_system(self):

        # # CONFIGURE OXYGEN TRANSPORT
        self.bed = NetworkAutoregulation(
            None, 
            self.tend, 
            None,
            self.ncomm,
            io=self.io
        )

        # # CONFIGURE CELLULAR METABOLISM
        if ( self.ncomm is not None ) and ( self.ncomm.Get_rank() != 0 ):
            self.mus = None

        else:
            self.mus = SkeletalMuscle(
                None, 
                self.tend, 
                None,
                self.mcomm,
                io=self.io
            )   

        # # STORE PARAMETERS
        self.t  = 0
        self.dt = min( [self.bed.dt, self.mus.dt] ) # global time step size
        self.Nt = round( self.tend / self.dt )      # global number of steps

        self.bed.step_increment = self.bed.dt / self.dt
        self.mus.step_increment = self.mus.dt / self.dt


    def time_evolution(self, show_progress=False):

        pbar =  tqdm(
            range(self.Nt), 
            desc='Processing', 
            disable=not show_progress
        )

        models = dict(zip(
            self.models, 
            [self.bed.network, self.bed.tone, self.mus.O2, self.mus.CM]
        ))

        for step in pbar:

            # # BOUNDARY CONDITIONS
            self.bed.update_root_condition()
            self.bed.update_terminal_condition()
            self.mus.update_atp_utilisation()

            # # SOLVE COUPLED SYSTEM
            self.step(step)

            # # UPDATE TIME
            self.t     += self.dt
            self.bed.t += self.bed.dt
            self.mus.t += self.mus.dt

            # # WRITE SOLUTION
            self.io.write_solutions(step, self.t, **models)


    def step(self, step):

        # # SOLVE NETWORK PRESSURE
        if self.bed.step_override > 0:

            # ? solve
            self.bed.network.solve()

            # ? compute coupling parameters
            self.bed.network.compute_vessel_parameters()
            avg_tension = self.bed.network.compute_tension_average(self.groups)
            avg_shear   = self.bed.network.compute_shear_average(self.groups)

            # ? compute total flow at root
            flow = self.bed.network.vessels[0].q + self.bed.network.vessels[1].q

            # ? convert units
            avg_tension = avg_tension * 1000    # N/m  to dyn/cm
            avg_shear   = avg_shear   * 10      # N/m2 to dyn/cm2
            flow        = flow        * 60000   # m3/s to L/min


        # # SOLVE SKELETAL MUSCLE
        if self.mus is not None:

            # ? update muscle flow
            self.mus.O2.update_flow(flow)

            # ? solve
            self.mus.O2.solve()
            self.mus.CM.solve()

            # ? compute rate of oxphos
            self.mus.update_oxygen_exchange()

            # ? free capillary oxygenation
            o2 = self.mus.o2cap

            # ? update local time
            self.mus.t += self.mus.dt


        # # SOLVE TONE REGULATION SYSTEM
        if self.bed.tone is not None:

            # ? update tension
            self.tension = avg_tension

            # ? compute smooth muscle tone
            self.muscle_tone(avg_tension, avg_shear, o2)

            # ? solve
            self.tone.solve()

            # ? update local time
            self.bed.t += self.bed.dt


        # ? update network geometry
        self.bed.update_network_tone()

        # # UPDATE TIME
        self.t     += self.dt


    def update_muscle_tone(self, tension, shear, o2):
        tone = ( 
            ( self.cmyo * tension ) - ( self.ctau * shear ) - ( self.cmet * o2 )
        )
        self.bed.update_tone(tone)



        