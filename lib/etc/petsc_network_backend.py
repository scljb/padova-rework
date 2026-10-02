"""
auth: L. J. Barratt
date: 03/11/25
"""

#%% PACKAGES

#* EXTERNAL LIBRARIES
from petsc4py import PETSc

INSERT = PETSc.InsertMode.INSERT_VALUES
ADD    = PETSc.InsertMode.ADD_VALUES


#%% CLASS

class PETScNetworkSolver():
    '''
    Class housing MPI-aware methods to solve Hagen-Poiseuille flow in
    an asymmetric fractal using a PETSc-based linear solver for steady
    -state pressure and flow in the microvascular network.

    Parameters
    ----------
    network : MicrovascularNetwork
        The vascular network object, providing vessel geometry, 
        connectivity, fluid properties, and MPI communicator.
    '''

    def __init__(self, network):
        '''
        Initialises a PETSc solver for the given network.

        Stores references to the network and its MPI communicator, an 
        initialises the dirty flags for the stiffness matrix and load
        vector.

        Parameters
        ----------
        network : MicrovascularNetwork
            The vascular network object, providing vessel geometry, 
            connectivity, fluid properties, and MPI communicator.
        '''

        self.net   = network
        self.comm  = self.net.comm
        self.rank  = self.net.comm.Get_rank()

        # a flag for if the matrix/vector has been changed
        # needs to be set true whenever setValues() is used
        self._A_dirty = True
        self._b_dirty = True


    def initialise_system(self):
        '''
        Initialises the full linear system ready for solving.
        '''

        self.comm.Barrier()

        # self.print('Creating PETSc objects...')
        self.create_petsc_objects()

        # self.print('Setting up solver...')
        self.setup_solver()

        # self.print('Initialising A matrix...')
        self.update_matrix()
        self.assemble_matrix()

        # self.print('Initialising b vector...')
        self.update_inlet_condition()
        self.update_outlet_condition()

        # final assembly
        self.assemble_matrix()
        self.assemble_vector()
        self.ksp.setOperators(self.A)

    
    def create_petsc_objects(self):    
        '''
        Initialises the PETSc matrix and vector objects for the linea
        system.

        The system size N is set to the number of vessels plus one. 
        Creates a sparse AIJ matrix A of size NxN, preallocated with
        up to 6 non-zeros per row, and an MPI-distributed load vector 
        b. A duplicate of b is created as the solution vector p 
        (nodal pressures).
        '''  

        #* MATRIX SIZE
        self.N = len(self.vessels) + 1

        #* BUILD PETSc MATRIX
        self.A = PETSc.Mat().createAIJ(
            size=(self.N, self.N),
            comm=self.comm
        )
        self.A.setPreallocationNNZ(10)
        # self.A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.A.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
        self.A.setUp()

        # b vector
        self.b = PETSc.Vec().createMPI(self.N, comm=self.comm)
        self.p = self.b.duplicate()     # pressure array

    
    def assemble_matrix(self):
        '''
        Finalises assembly of the PETSc stiffness matrix and marks 
        it as clean.
        '''
        
        self.A.assemblyBegin()
        self.A.assemblyEnd()        
        self._A_dirty = False

    
    def assemble_vector(self):
        '''
        Finalises assembly of the PETSc load vector and marks it as 
        clean.
        '''
        
        self.b.assemblyBegin()
        self.b.assemblyEnd()
        self._b_dirty = False
        

    def update_matrix(self, addv=ADD):
        '''
        Updates the stiffness matrix with current vessel conductance 
        values.

        Zeros all existing entries and repopulates the matrix using 
        the conductance (g) of each vessel. Marks the matrix as dirty 
        so it is reassembled before the next solve.
        '''

        if not self._A_dirty:
            self.A.zeroEntries()

        # rank owndership
        rstart, rend = self.A.getOwnershipRange()

        # loop over list of vessels
        for v in self.vessels:

            i = v.i
            j = v.j
            g = v.g

            if rstart <= i < rend:
                self.A.setValue(i, i,  g, addv=addv)
                self.A.setValue(i, j, -g, addv=addv)

            if rstart <= j < rend:
                self.A.setValue(j, j,  g, addv=addv)
                self.A.setValue(j, i, -g, addv=addv)


        self._A_dirty = True

    
    def setup_solver(self):
        '''
        Configures the PETSc KSP linear solver.

        Sets up a Conjugate Gradient (CG) solver with a HYPRE 
        preconditioner. 
        
        Options can be overridden at runtime via PETSc's command-line 
        options system (setFromOptions).
        '''

        self.ksp = PETSc.KSP().create(comm=self.comm)

        self.ksp.setOperators(self.A)

        # solver type
        self.ksp.setType('cg')

        # preconditioner
        pc = self.ksp.getPC() # ! this might not be used?
        pc.setType('hypre')

        # tolerances
        self.ksp.setTolerances(rtol=1e-12)

        self.ksp.setFromOptions()


    def update_inlet_condition(self, new_root_value=None, addv=INSERT):
        '''
        Updates the inlet (root) boundary condition applied to the 
        linear system.

        If the stiffness matrix is dirty, it is assembled first with 
        a warning, as boundary conditions must be applied to a clean 
        matrix. If new_root_value is provided, it overwrites
        the network's root value.

        The boundary condition type is determined by net.inflow_type 
        (boolean):
            - True:  Neumann    - the root flow value is 
                                  inserted directly into the 
                                  load vector.
            - False: Dirichlet  - the root row is zeroed in the 
                                  stiffness matrix and the root 
                                  pressure is set in the load vector.

        Marks the load vector as dirty so it is reassembled before 
        the next solve.

        Parameters
        ----------
        new_root_value : float or None
            New inlet flow (Neumann) or pressure (Dirichlet) value 
            to apply.
            If None, the existing root value is used.
        '''

        if self._A_dirty:
            # self.print(
            #     'WARNING: A matrix must be \'clean\' '      \
            #     'prior to updating boundary conditions. '   \
            #     'Calling assemble_matrix()... '
            # )

            self.assemble_matrix()


        if new_root_value is not None:
            self.net.root = new_root_value


        root_idx = 0

        # Neumann (flow) condition
        if self.net.inflow_type:
            self.b.setValue(root_idx, self.net.root, addv=addv)
            

        # Dirichlet (pressure) condition
        else:
            self.A.zeroRows([root_idx], diag=1.0)
            self.b.setValue(root_idx, self.net.root, addv=addv)
            self.ksp.setOperators(self.A)


        self._b_dirty = True

    
    def update_outlet_condition(self, new_terminal_value=None):
        '''
        Updates the outlet (terminal) boundary condition applied to 
        the linear system.

        If new_terminal_value is provided, it overwrites the 
        network's terminal pressure.
        
        The Dirichlet condition is enforced by zeroing the 
        corresponding rows in the stiffness matrix and setting the 
        terminal pressure value in the load vector.
        
        Marks the load vector as dirty so it is reassembled before 
        the next solve.

        Parameters
        ----------
        new_terminal_value : float or None
            New terminal pressure to apply. If None, the existing 
            terminal pressure is used.
        '''

        if self._A_dirty:
            self.assemble_matrix()

        if new_terminal_value is not None:
            self.net.terminal = new_terminal_value

        # Dirichlet (pressure) condition
        idx = self.net.terminals
        self.A.zeroRows(idx, diag=1.0)
        self.b.setValues(idx, [self.net.terminal] * len(idx))
        self.ksp.setOperators(self.A)     
        self._b_dirty = True


    def update_vessel_properties(self, lmbda, groups=None):
        '''
        Calls MicrovascularNetwork method to update vessel 
        properties according current state of dilation.
        
        After updating vessel properties, the system matrix 
        is rebuilt to reflect the new conductance values.

        Parameters
        ----------
        groups : list[str] or None
            Group names to average over. If None, averages over the
            whole network. 
        '''

        #* update properties in vessel list
        self.net.update_vessel_list(lmbda, groups=groups)

        #* update matrix
        self.update_matrix()


    def solve(self):
        '''
        Solves the linear system Ap = b using a KSP solver.

        Assembles the stiffness matrix and/or load vector if they are 
        marked as dirty (i.e. out of date). Raises a RuntimeError if 
        the KSP solver fails to converge.
        '''

        if self._A_dirty:
            self.assemble_matrix()

        if self._b_dirty:
            self.assemble_vector()
            

        self.ksp.solve(self.b, self.p)

        # check convergence
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f'KSP did not converge: {reason}')
        

    def get_pressure_array(self, addv=ADD):
        '''
        Gather all of the pressure solution array segments to Rank 0.

        Other ranks return None.
        '''

        scatter, p_full = PETSc.Scatter.toZero(self.p)
        scatter.scatter(self.p, p_full, addv=addv)
        
        if self.rank == 0:
            return p_full.getArray()
        
        return None


    @property
    def vessels(self):
        return self.net.vessels
    

