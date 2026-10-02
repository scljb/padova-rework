"""
auth: L. J. Barratt
Created on 03/11/25

Class housing methods to solve Hagen-Poiseuille flow in an asymmetric fractal
using the PARDISO solver library.
"""

#%% PACKAGES

#* EXTERNAL LIBRARIES
from numpy import sqrt, pi, empty, int32, int64, zeros, mean, nan, arange,     \
                  array, fromiter, repeat, tile, setdiff1d, add, isin, where,  \
                  full
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import pypardiso
import time

#* INTERNAL LIBRARIES


#%% CLASS
class PARDISONetworkSolver():

    def __init__(self, network):
        self.net = network


    def initialise_system(self):

        # print('Initialising A matrix...')
        self.assemble_matrix()

        # print('Initialising b vector...')
        self.assemble_vector()

        # print('Creating index maps...')
        self.build_index_maps()


    def assemble_matrix(self):
        """
        Builds the A and b tensors for the algebraic solver.
        """

        # Vector size
        self.N = len(self.vessels) + 1
        m      = 4 * len(self.vessels)

        # Vector init for sparse matrix
        rows = empty( m, dtype=int32 )      # stores the row position (i)
        cols = empty( m, dtype=int32 )      # stores the column position (j)
        data = empty( m, dtype=float )      # stores the coefficient (g)

        # Assign coefficients to the vectors
        for idx, ves in enumerate(self.vessels):

            #? location based on node index
            b = 4 * idx

            #? each vessel becomes a 2x2 matrix
            rows[b:b+4] = [ ves.i, ves.j,  ves.i,  ves.j ]
            cols[b:b+4] = [ ves.i, ves.j,  ves.j,  ves.i ]
            data[b:b+4] = [ ves.g, ves.g, -ves.g, -ves.g ]


        # Form matrix
        self.A = csr_matrix((data, (rows, cols)), shape=(self.N, self.N))
        self.b = zeros( self.N, dtype=float )

        # Store matrix data for updates later
        self._rows = rows
        self._cols = cols
        self._data = data
        self._A_base = self.A.copy()  # immutable variant of the initial conditions


    def assemble_vector(self):
        # Define boundary conditions in a vector
        #? root condition
        if self.net.inflow_type:    # Neumann Flow condition
            bcs = { }

        else:                   # Dirichlet Pressure condition
            bcs = { 0: self.net.root }

        #? terminal conditions, dirichlet
        for node in self.net.terminals:
            bcs[node] = self.net.terminal


        # Identify fixed (boundary, i.e. known pressure) and free (internal) nodes
        self.bndrynodes = sorted(bcs.keys())
        self.intnlnodes = setdiff1d(arange(self.N), self.bndrynodes)
        self.bndrypress = array([bcs[k] for k in self.bndrynodes], dtype=float)


    def build_index_maps(self):
        """
        Build mapping from each vessel contribution to the correct A.data index.
        Also creates per-group maps (arteries, arterioles, capillaries).

        #! this currently creates group idx maps that are not used anywhere.
        """

        # Build (row, col) -> data-index map
        coo = self.A.tocoo()
        pos = {(r, c): k for k, (r, c) in enumerate(zip(coo.row, coo.col))}

        # Prepare outputs
        m = len(self.vessels)
        self._idx_map = empty(4 * m, dtype=int64)

        # if vessel_groups exist, precompute reverse lookup
        group_names = list(getattr(self.net, "vessel_groups", {}).keys())
        n_groups = len(group_names)
        group_of_vessel = full(m, -1, dtype=int)
        if n_groups > 0:
            for gi, vids in enumerate(self.net.vessel_groups.values()):
                group_of_vessel[vids] = gi
            group_entries = [[] for _ in range(n_groups)]

        # Single pass over all vessels
        for idx, v in enumerate(self.vessels):
            b = 4 * idx
            pairs = [(v.i, v.i), (v.j, v.j), (v.i, v.j), (v.j, v.i)]
            self._idx_map[b:b+4] = [pos[(r, c)] for (r, c) in pairs]

            # add to that vessel's group map
            if n_groups > 0:
                gi = group_of_vessel[idx]
                if gi != -1:
                    group_entries[gi].extend(self._idx_map[b:b+4])

        # Finalize
        self.group_idx_maps = {
            group_names[i]: array(group_entries[i], dtype=int64)
            for i in range(n_groups)
        }

        # ! prints
        # for name, arr in self.group_idx_maps.items():
        #     print(f'\t{name}\t: {len(arr)} entries mapped')


        # print(f'\ttotal\t\t: {len(self._idx_map)} entries mapped')


    def update_inlet_condition(self, new_root_value=None):
        """
        Update the inlet (root) boundary value in A and b.
        """

        if new_root_value is not None:
            self.net.root = new_root_value

        # The root index
        root_idx = 0

        if self.net.inflow_type:    # flow condition
            self.b[root_idx] = self.net.root

        else:
            # Update the boundary pressure array
            self.bndrypress[self.bndrynodes.index(root_idx)] = self.net.root

            # Apply the Dirichlet condition directly to A and b
            self.A.data[self.A.indptr[root_idx]:self.A.indptr[root_idx+1]] = 0.0
            self.A[root_idx, root_idx] = 1.0
            self.b[root_idx] = self.net.root


    def update_outlet_condition(self, new_terminal_pressure=None):
        """
        Update all terminal (outlet) boundary pressures in A and b.
        """

        # Store new terminal pressure
        if new_terminal_pressure is not None:
            self.net.terminal = new_terminal_pressure

        # Identify terminal node indices
        term_nodes = self.net.terminals  # list/array of outlet nodes

        # Update the stored boundary pressures
        self.bndrypress[isin(self.bndrynodes, self.net.terminals)] = self.net.terminal

        # Apply Dirichlet BCs directly to A and b for each terminal node
        for node in term_nodes:
            # Zero out row corresponding to terminal node
            row_start = self.A.indptr[node]
            row_end   = self.A.indptr[node + 1]
            self.A.data[row_start:row_end] = 0.0

            # Set diagonal entry to 1
            self.A[node, node] = 1.0

            # Set right-hand side to terminal pressure
            self.b[node] = self.net.terminal


    def update_vessel_properties(self, lmbda, groups=None):
        '''
        Calls MicrovascularNetwork method to update vessel 
        properties according current state of dilation.
        
        After updating vessel properties, the system matrix 
        is rebuilt to reflect the new conductance values.

        Parameters
        ----------
        groups : list[str] or None
            Group names of vessels to be updated. If None, 
            all vessels are updated.
        '''

        #* update properties in vessel list
        self.net.update_vessel_list(lmbda, groups=groups)

        #* update matrix 
        garr    = fromiter(
            (v.g for v in self.vessels), 
            dtype=float, 
            count=len(self.vessels)
        )
        contrib = repeat(garr, 4) * tile([1,1,-1,-1], len(garr))

        self.A.data[:] = 0.0
        add.at(self.A.data, self._idx_map, contrib)


    def solve(self):
        '''
        Solves algebraic system for nodal pressure.
        '''

        # A_ff = self.A[self.intnlnodes[:, None], self.intnlnodes]
        # A_fd = self.A[self.intnlnodes[:, None], self.bndrynodes]

        # Construct tensors
        A_int = self.A[self.intnlnodes]
        A_ff  = A_int[:, self.intnlnodes]
        A_fd  = A_int[:, self.bndrynodes]
        b_f   = self.b[self.intnlnodes] - A_fd @ self.bndrypress

        # Solve for the internal nodes
        p_free = pypardiso.spsolve(A_ff, b_f)

        # Contruct full pressure solution vector
        if not hasattr(self, 'p'):
            self.p = zeros(self.N)

        self.p[self.bndrynodes] = self.bndrypress
        self.p[self.intnlnodes] = p_free


    def compute_tree_compliance(self):
        """
        Compute the small-signal effective compliance of the microvascular tree:
            C_eff = dV_tree / dP_in   (terminals held at fixed Dirichlet pressure)

        Assumes each vessel has already been assigned a stiffness v.beta
        using the relation: beta = sqrt(pi) * Eh / (1 - nu^2).

        Parameters
        ----------
        P_ext : float
            External pressure for transmural pressure (default = 0).

        Returns
        -------
        C_eff : float
            Total inlet compliance (m^3/Pa)
        C_groups : dict (optional)
            Per-group compliances, summed by vessel groups.
        """

        if not hasattr(self, "p"):
            raise AttributeError("Run solve() before computing compliance.")


        #* Build the pressure sensitivity solve
        root_idx = 0  # your inlet node

        bndry = array(self.bndrynodes, dtype=int)
        free  = array(self.intnlnodes, dtype=int)

        # Dirichlet sensitivity values: inlet=1, terminals=0
        delta_p_bnd = zeros(len(bndry), dtype=float)
        inlet_pos   = where(bndry == root_idx)[0]
        delta_p_bnd[inlet_pos[0]] = 1.0

        # Partition matrix
        A_ff = self.A[free[:, None], free]
        A_fd = self.A[free[:, None], bndry]

        # RHS
        rhs = -A_fd @ delta_p_bnd

        # Solve for Δp on free nodes
        delta_p_free = spsolve(A_ff, rhs)

        # Full Δp vector
        delta_p        = zeros(self.N, dtype=float)
        delta_p[bndry] = delta_p_bnd
        delta_p[free]  = delta_p_free


        #* Compute vessel volume compliance from the tube law
        nE = len(self.vessels)
        c_edge = zeros(nE, dtype=float)   # vessel compliance [m^3/Pa]

        for k, v in enumerate(self.vessels):

            A_k   = pi * v.r**2         # area
            dA_dP = 2.0 * sqrt(A_k) / v.b

            c_edge[k] = dA_dP * v.L     # compliance = (dA/dP)*L


        #* Compute total compliance by weighting per-vessel compliance by pressure sensitivity
        C_eff = 0.0
        for k, v in enumerate(self.vessels):
            dp_bar = 0.5 * (delta_p[v.i] + delta_p[v.j])
            C_eff += c_edge[k] * dp_bar


        #* Group compliance (arteries, arterioles, capillaries)
        # C_groups = {}
        # if hasattr(self, "vessel_groups"):
        #     for name, ids in self.vessel_groups.items():
        #         Cg = 0.0
        #         for k in ids:
        #             v = self.vessels[k]
        #             dp_bar = 0.5 * (delta_p[v.i] + delta_p[v.j])
        #             Cg += c_edge[k] * dp_bar
        #         C_groups[name] = Cg


        return C_eff #, C_groups
    

    def get_pressure_array(self):
        return self.p
    

    @property
    def vessels(self):
        return self.net.vessels
