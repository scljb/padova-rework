"""
auth: L. J. Barratt
Created on 12/05/25

Cass housing methods for generating an arterial network.
"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
from os     import makedirs
from numpy  import sqrt
import sys

# * INTERNAL LIBRARIES
from lib.solvers.base_solver import BaseSolver
from lib.models.artery  import Artery
from lib.etc            import fem_solvers as fems
from lib.etc.artery_utillities import inflow_rate_adaptive as root_solver,  \
                                      splitting as junction_solve,          \
                                      check_stability


#%% CLASS

class ArterialNetwork(BaseSolver):

    models = ['Artery']

    def configure_mpi(self, comm):

        super().configure_mpi(comm)

        if comm is not None:
            from lib.mpi.artery_mpi import ArteryMPI
            self.mpi = ArteryMPI(comm)

        else:
            self.mpi = None


    def create_system(self):

        io = self.io.model_io['Artery']

        # # STORE PARAMETERS        
        # * Solver options
        self.solver_params = fems.lu  
        # ! mumps, {lu}, gmres, bicgstab
        
        # * Time Parameters
        self.t    = 0                               # initial time [s]
        self.dt   = io.solution['dt'][0]            # time step size [s]
        self.Nt   = int(round(self.tend/self.dt))+1 # number of steps

        # * Network Parameters
        # number of arteries in network
        self.nart      = int(io.network['number_of_arteries'][0])

        # list of arteries by id 
        # ? this will hold the Artery instances
        self.arteries  = list(range(self.nart))

        # list of root and terminal ids
        self.roots     = io.network['roots']
        self.terminals = io.network['terminals']

        # dictionary mapping parent to daughter ids
        self.parents   = io.network['parents']

        # reverse mapping of daughters to parents
        self.daughters = {}
        for p, daughters in self.parents.items():
            for d in daughters:
                self.daughters.setdefault(d, []).append(p)


        # # RANK DISTRIBUTION
        if self.mpi is not None:
            # distribute arteries to ranks
            self.local_arteries, rank_to_arteries = \
                    self.mpi.distribute_arteries(self.nart)

        else:
            indices = list(range(len(self.arteries)))
            self.local_arteries = indices
            rank_to_arteries    = {0: indices}

        
        # map artery ids to owner rank
        self.arteries_to_rank = {
            a: r
            for r, arteries in rank_to_arteries.items()
            for a in arteries
        }

        # store owned artery ids by groups
        self.local_roots     = []
        self.local_parents   = {}
        self.local_daughters = {}
        self.local_terminals = []

        for idx in self.local_arteries:

            if idx in self.roots:
                self.local_roots.append(idx)

            if idx in self.parents:
                self.local_parents[idx] = self.parents[idx]

            if idx in self.daughters:
                self.local_daughters[idx] = self.daughters[idx]

            if idx in self.terminals:
                self.local_terminals.append(idx)


        # # CREATE NETWORK
        self.generate_network()


    def generate_network(self):

        io = self.io.model_io['Artery']

        for idx in self.local_arteries:

            # ? vessel type
            is_root     = idx in self.roots
            is_parent   = idx in self.parents
            is_terminal = idx in self.terminals

            if not (is_root or is_parent or is_terminal):
                raise ValueError(f'Artery {idx} cannot be classified '
                                  'as root, parent, or terminal.')

            # ? junction type: pass-through or split/merge
            is_pass = is_parent and ( len( self.parents[idx] ) == 1 )

            # ? get terminal index
            terminal_idx = None
            if is_terminal:
                terminal_idx = self.terminals.index(idx)


            # ? create per artery comm for arterial instance
            if self.mpi is not None:
                comm_self = self.mpi.comm.Split(color=self.mpi.rank, key=0)

            else: 
                from mpi4py.MPI import COMM_SELF
                comm_self = COMM_SELF


            # ? create instance
            self.arteries[idx] = Artery(
                idx,
                terminal_idx,
                io.fluid,
                io.vessel,
                io.outflow,
                io.solution,
                is_root,
                is_terminal,
                is_pass,
                comm_self
            )

            # ? define finite element system
            self.arteries[idx].define_domain()
            self.arteries[idx].define_variational_problem(
                self.solver_params
            )

            # ? create folder for output
            output = self.io.output_dir / f'artery_{idx}'
            makedirs(output, exist_ok=False)


    def initialise_solver(self):
        # self.m_attrs = self.get_model_dict(bed=self.bed, mus=self.mus)
        pass


    def time_evolution(self, pbar):

        for step in pbar:

            pbar.set_postfix(t=self._postfix())

            # # SOLVE COUPLED SYSTEM
            self.step(step, write=True)


    def step(self, step, write=False):

        # # UPDATE ROOT CONDITION
        self.update_root_condition(step)

        # # SOLVE EACH ARTERIAL SYSTEM
        for idx in self.local_arteries:
        
            # ? solve
            artery = self.arteries[idx]
            artery.solve()

            # ? output to file
            # Output on the first and final (firnal) step plus every write step
            firnal = ( step==0 ) or ( step==self.Nt )
            if ( firnal or not step % self.io.write_increment ) and write:
                self.io.model_io['Artery'].write_arterial_solution(
                    artery, self.t
                )

            # ? update old solution
            artery.update_solution()

            # ? check stability
            check_stability(idx, artery, self.dt, disp=False)


        # # UPDATE JUNCTION & TERMINAL CONDITIONS
        self.update_junction_condition()
        self.update_terminal_condition(step)

        # # UPDATE TIME
        self.t += self.dt


    def update_root_condition(self, step):
        '''
        Updates the root inlet condition for the network.

        Compute the unique state (As, Us) using appropriate function.
        Then compute the velocity at a 'ghost' node outside the domain,
        applying this new value to the root, assuming Ar=Al.
        '''

        for ii in self.local_roots:

            # ? compute current value
            Qin = self.rc.compute_value(self.t)

            # ? call newton-raphson solver to compute A and U for given Qin
            As0, Us0, Ur = root_solver(self.arteries[ii], Qin)

            # ? calculate velocity of ghost node
            Ul = 2*Us0 - Ur     

            # ? apply condition
            self.arteries[ii].inlet.assign(Ul) # area is assumed zero-gradient


    def update_junction_condition(self):

        for pid, dids in self.local_parents.items():
                        
            p_rank  = self.arteries_to_rank[pid]
            d_ranks = [self.arteries_to_rank[d] for d in dids]

            # ? check if the current rank is involved in this junction
            if not ( (self.rank == p_rank) or (self.rank in d_ranks) ):
                return

            # ? ranks owning the parent: receive -> solve -> send -> apply
            if self.rank == p_rank:
                self.junction_parent_solver(pid, dids, d_ranks)
                return 

            # ? ranks owning a daughter: send -> receive -> apply
            for did, rid in zip(dids, d_ranks):
                #print(did)  # ! check if this is actually a scalar or vector
                if self.rank == rid:
                    self.junction_daughter_helper(self.arteries, did, p_rank)


    def junction_parent_solver(self, pid, dids, d_ranks):

        dx = self.arteries[pid].dx

        # # GET INFLOW SOLUTIONS FOR DAUGHTER(S)
        daughters = [None] * len(dids)
        for i, (did, rid) in enumerate( zip(dids, d_ranks) ):

            if rid == self.rank:
                # Local daughter
                daughters[i] = (
                    self.arteries[did].probe_spatial_solutions(dx),
                    self.arteries[did].probe_spatial_properties(0.0)
                )

            else:
                # Remote daughter
                daughters[i] = self.mpi.comm.recv(source=rid, tag=did)


        # # GET OUTFLOW DATA FOR PARENT
        L = self.arteries[pid].L
        parent = (
            self.arteries[pid].probe_spatial_solutions(L-dx),
            self.arteries[pid].probe_spatial_properties(L)
        )

        # # SOLVE JUNCTION
        solution = junction_solve(
            parent, 
            *daughters,
            ids=(pid, *dids)
        )
        p, d1, d2 = solution

        # # APPLY SOLUTIONS
        # ? apply area solution to parent
        As = p[0]
        self.arteries[pid].outlet.assign(As)

        # ? apply velocity solutions to daughter(s)
        daughter_velocities = [ d1[1], d2[1] ]
        for vel, did, rid in zip(daughter_velocities, dids, d_ranks):
            # velocity is None, the daughter does not exist
            if vel is None:
                continue

            if rid == self.rank:
                # Local apply
                self.arteries[did].inlet.assign(vel)
            else:  
                # Send to daughter
                self.comm.send(vel, dest=rid, tag=did)


    def junction_daughter_helper(self, did, p_rank):
        '''
        Only used in parallelised networks.
        '''

        dx = self.arteries[did].dx

        # # GET ROOT BOUNDARY VARIABLES & SEND TO PARENT
        data = (
            self.arteries[did].probe_spatial_solutions(dx),
            self.arteries[did].probe_spatial_properties(0.0)
        )
        self.comm.send(data, dest=p_rank, tag=did)

        # # RECIEVE UPDATED SOLUTION
        vel = self.comm.recv(source=p_rank, tag=did)

        # # APPLY SOLUTION
        self.arteries[did].inlet.assign(vel)


    def update_terminal_condition(self, step):
        '''
        Boundary velocity defined by the defined reflection (Rt) of the forward propagating wave.
        '''

        for ii in self.local_terminals:

            artery = self.arteries[ii]

            # ? boundary values
            Al = artery.Un(artery.L,)[0]
            Ul = artery.Un(artery.L,)[1]

            # ? compute new boundary solution
            # wave speed
            c  = pow(artery.beta/2/artery.rho/artery.A0o,0.5) * pow(Al,0.25)

            # forward propagating wave
            Wf = Ul - artery.u0 + 4 * ( c - artery.c0o )

            # Compute ghost node velocity
            Ur = Wf * ( 1 - artery.Rt ) - Ul

            # ? apply condition
            artery.outlet.assign(Ur)  # Area is assumed zero-gradient


    def compute_junction_variables(self, artery, is_parent):
        '''
        Compute the solution at the foot of the outgoing characteristic.
         
        Returns these solutions and the boundary properties.

        # ! Needs further testing. the application of the result seems to break this method. if i could apply the boundary conditions though the characteristics, this may work better. or maybe dampen the step between the previous solution and the applied condition.
        # ! or, more likely, compute a more appropriate A/U after the junction solve using characteristics (linear extrapolation?) 
        '''

        if is_parent:
            xb = artery.L
        else:
            xb = 0.0


        prop  = artery.probe_spatial_properties(x=xb)
        s_end = artery.probe_spatial_solutions(x=xb)

        # characteristic speed at the junction end
        c   = sqrt( prop['beta'] / (
                2.0 * prop['rho'] * prop['A0']
            )
        ) * s_end['An']**0.25

        if is_parent:
            lam = c + s_end['Un']
        else:
            lam = s_end['Un'] - c

        d = min(max(lam * self.dt, 0.0), artery.dx)

        if is_parent:
            x_foot = artery.L - d
        else:
            x_foot = d

        return ( artery.probe_spatial_solutions(x=x_foot), prop )