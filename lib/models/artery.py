"""
auth: L. J. Barratt
Created on 07/05/25

Class housing methods to solve 1D Navier-Stokes solver for flow in compliant tubes using FEniCS.
"""

#%% PACKAGES

# * EXTERNAL LIBRARIES
from fenics import IntervalMesh, FiniteElement, FunctionSpace, Expression,  \
                   Measure, Function, split, TestFunction, near,            \
                   Constant, DirichletBC, derivative, grad, DOLFIN_EPS,     \
                   NonlinearVariationalProblem, NonlinearVariationalSolver
                #    as_vector, as_matrix, plot, interpolate, project
from numpy import pi, log, sqrt, ceil, array

sol_dtype = [
    ('An', 'd'), 
    ('Un', 'd')
]

prop_dtype = [
    ('A0', 'd'), 
    ('u0', 'd'),
    ('beta', 'd'), 
    ('rho', 'd')
]

# ! i added mpi_comm=self.comm to all Expressions I could find. might cause problems?


#%% CLASS

class Artery:

    def __init__(
            self, 
            ii, jj, 
            fluid, vessel, outflow, sol, 
            is_root, is_terminal, pass_through,
            comm
        ):

        if comm is not None:
            self.comm = comm

        else:
            raise ValueError('no comm, no run')
        

        self.artery_id = ii
        self.pass_through = pass_through

        # # SOLUTION PARAMETERS
        self.dt     = sol['dt'][0]          # Time Step [s]
        self.plyodr = int(sol['plyodr'][0]) # Poly. order of finite elements

        self._variational_form  = self.crank_nicholson_scheme
        self.theta  = sol['theta'][0]       # Crank-Nicholson parameter [-]

        # # FLUID PARAMETERS
        self.u0    = fluid['u0'][0]     # Initial condition for velocity [m/s]
        self.gamma = fluid['gamma'][0]  # Flow Parameter [-]
        self.rho   = fluid['rho'][0]    # Fluid Density [kg/m3]
        self.pd    = fluid['pd'][0]     # Distal Pressure [Pa]
        mu         = fluid['mu'][0]     # Fluid Viscosity [Pa s]

        # ? flow resistance coefficient
        self.Kr = 2*(self.gamma+2)*pi*mu/self.rho

        # # VESSEL MATERIAL
        E  = vessel['e'][ii]            # Young's Modulus [Pa]
        h  = vessel['h'][ii]            # Vessel Thickness [m]
        nu = vessel['nu'][0]            # Poisson's Ratio [-]

        # ? material coefficient [kg/s2/cm2]
        self.beta = E*h*pow(pi,0.5)/(1-nu**2)

         # # VESSEL GEOMETRY
        self.L  = vessel['l'][ii]       # Vessel Length [m] 
        self.dx = float(sol['dx'][0])   # Cell size [m]

        # ? store the rest in a dict for later
        keys = [
            'rmax', 'rmin', 'state', 'c', 'x0', 'kappa', 'delta', 'a1', 'a2'
        ]
        self.vessel = {}
        for key, values in vessel.items():
            if key not in keys:
                continue

            try:
                self.vessel[key] = float(values[ii])

            except ValueError:
                self.vessel[key] = values[ii]


        # # OUTFLOW PARAMETERS
        self.is_root     = is_root
        self.is_terminal = is_terminal

        if self.is_terminal:
            self.outflow_model = outflow['type'][jj]

            if self.outflow_model == 'reflective':
                self.Rt = outflow['rt'][jj]

        else: 
            self.outflow_model = 'junction'


    def define_domain(self):

        # # CREATE MESH
        Nx = int( ceil( self.L / self.dx ) )
        self.mesh = IntervalMesh(self.comm, Nx, 0, self.L)

        # # DEFINE FINITE ELEMENT TYPE
        El = FiniteElement('CG', self.mesh.ufl_cell(), self.plyodr)

        # # DEFINE FUNCTION SPACES
        # ? Mixed scaler space: area, velocity
        self.V = FunctionSpace(self.mesh, El*El)

        # ? Scaler space: pressure
        self.W = FunctionSpace(self.mesh, El)

        # # DEFINE GEOMETRY & MATERIALS
        # ? variables
        rmax = self.vessel['rmax']
        rmin = self.vessel['rmin']

        # Reference Radius
        if self.vessel['state'] == 'healthy':
            deg = 3

            r0  = Expression(
                    'Rp * pow(Rd/Rp, x[0]/L)', 
                    degree=deg,
                    Rp=rmax, 
                    Rd=rmin, 
                    L=self.L, 
                    mpi_comm=self.comm
            )
            
            dr0 = Expression('logRdRp / L * r0', 
                    degree=deg, 
                    logRdRp=log(rmin/rmax), 
                    L=self.L, 
                    r0=r0, 
                    mpi_comm=self.comm
            )

            self.beta0 = Expression(
                'beta0', degree=1, beta0=self.beta, mpi_comm=self.comm
            )

            self.dbeta0 = Expression(
                '0.0',
                degree=1, mpi_comm=self.comm
            )
            
        elif self.vessel['state'] == 'stenotic': 
            deg = 4

            alpha = 10000
            g_s  = ( alpha*self.vessel['c'] ) / self.L**4

            g   = Expression(
                    'exp(-g_s*pow(x[0]-x0,4))', 
                    degree=deg,
                    g_s=g_s, 
                    x0=self.vessel['x0'], 
                    mpi_comm=self.comm
            )
            
            dg  = Expression(
                    '-4*g_s*pow(x[0]-x0,3)*g', 
                    degree=deg, 
                    g_s=g_s, 
                    x0=self.vessel['x0'], 
                    g=g, 
                    mpi_comm=self.comm
            )
            
            r0  = Expression(
                    'Rmax - (Rmax - Rmin) * g', 
                    degree=deg, 
                    Rmax=rmax, 
                    Rmin=rmin, 
                    g=g, 
                    mpi_comm=self.comm
            )
            
            dr0 = Expression(
                    '-(Rmax - Rmin) * dg', 
                    degree=deg,
                    Rmax=rmax, 
                    Rmin=rmin, 
                    dg=dg, 
                    mpi_comm=self.comm
            )

            self.beta0    = Expression(
                "x[0] < a1 - delta ? beta0 : "
                "x[0] < a1 + delta ? beta0 + ( kappa*beta0 - beta0 ) * ( 3*pow((x[0]-a1+delta)/(2*delta),2) - 2*pow((x[0]-a1+delta)/(2*delta),3) ) :"
                "x[0] < a2 - delta ? kappa * beta0 : "
                "x[0] < a2 + delta ? beta0 + ( kappa*beta0 - beta0 ) * ( 1 - ( 3*pow((x[0]-a2+delta)/(2*delta),2) - 2*pow((x[0]-a2+delta)/(2*delta),3) ) ) :"
                "beta0",
                degree=3, beta0=self.beta, kappa=self.kappa, a1=self.a1, a2=self.a2, delta=self.delta, mpi_comm=self.comm
            )

            self.dbeta0 = Expression(
                "x[0] < a1 - delta ? 0.0 : "
                "x[0] < a1 + delta ? "
                "(kappa*beta0 - beta0) * (6*((x[0]-a1+delta)/(2*delta)) - 6*pow((x[0]-a1+delta)/(2*delta),2)) / (2*delta) : "
                "x[0] < a2 - delta ? 0.0 : "
                "x[0] < a2 + delta ? "
                "-(kappa*beta0 - beta0) * (6*((x[0]-a2+delta)/(2*delta)) - 6*pow((x[0]-a2+delta)/(2*delta),2)) / (2*delta) : "
                "0.0",
                degree=2, beta0=self.beta, kappa=self.kappa, a1=self.a1, a2=self.a2, delta=self.delta, mpi_comm=self.comm
            )
            
        else: 
            raise ValueError(f'\'{self.vessel['state']}\' is not a supported artery state.')
        
        # ? reference area
        self.A0  = Expression(
            'pi * pow(r0, 2)', degree=deg, r0=r0, mpi_comm=self.comm
        )
        self.dA0 = Expression(
            '2*pi*r0*dr0', degree=deg, r0=r0, dr0=dr0, mpi_comm=self.comm
        )


        # # INITIAL WAVE SPEED
        self.c0 = Expression(
            'pow( beta / 2 / rho, 0.5 ) / pow( A0, 0.25 )',
            degree=deg,
            beta=self.beta0,
            rho=self.rho,
            A0=self.A0, 
            mpi_comm=self.comm
        )
        
        # # INITIAL BOUNDARY STATES
        self.A0i = self.A0(0,)              # Inlet reference area
        self.A0o = self.A0(self.L,)         # Outlet reference area

        self.beta0i = self.beta0(0,)        # Inlet reference beta
        self.beta0o = self.beta0(self.L,)   # Outlet reference beta

        self.c0i = self.c0(0,)              # Inlet reference wave speed
        self.c0o = self.c0(self.L,)         # Outlet reference wave speed


    def define_variational_problem(self, solver_params):
        """
        Define the system of coupled equations using the chosen time scheme.
        """

        # # DEFINE MEASURE
        dx = Measure("dx", domain=self.mesh)

        # # NEW SOLUTION
        self.U = Function(self.V)
        A, u   = split(self.U)

        # # OLD SOLUTION
        self.Un = Function(self.V)
        An, un  = split(self.Un)

        # # INITIAL CONDITION
        self.Un.assign(
            Expression(
                ('A0', 'u0'), 
                degree=2, 
                A0=self.A0, 
                u0=self.u0, 
                mpi_comm=self.comm
            )
        )

        # # PRESSURE SOLUTION
        self.pn = Function(self.W)
        self.pn.assign(
            Expression('pext', degree=2, pext=self.pd, mpi_comm=self.comm)
        )

        # # TEST FUNCTIONS
        psi_a, psi_u = TestFunction(self.V)

        # # VARIATIONAL EQUATION            
        self.vp = self._variational_form(dx, A, u, An, un, psi_a, psi_u)

        # # BOUNDARY CONDITIONS
        def inlet_bdry(x, on_boundary):
            return on_boundary and near(x[0], 0, 1e-14)


        def outlet_bdry(x, on_boundary):
            return on_boundary and near(x[0], self.L, 1e-14)
        

        # * Proximal condition
        # ? all inflows are (currently) velocity-defined boundaries
        # TODO pressure boundaries will apply to the area
        self.inlet = Constant(self.u0)
        bc_inlet   = DirichletBC(self.V.sub(1), self.inlet, inlet_bdry)


        # * distal condition
        if self.is_terminal:
            
            if self.outflow_model == 'reflective':
                # ? outflow is reflective. 
                # ? velocity-defined boundary.

                self.outlet = Constant(self.u0)
                bc_outlet   = DirichletBC(
                    self.V.sub(1), self.outlet, outlet_bdry
                )
            
            else:
                # ? outflow is coupled to a wk model.  
                # ? area-defined boundary.
                        
                self.outlet = Constant(self.A0o)
                bc_outlet   = DirichletBC(
                    self.V.sub(0), self.outlet, outlet_bdry
                )

        else:
            # ? outflow is a junction. 
            # ? area-defined boundary.
                        
            self.outlet = Constant(self.A0o)
            bc_outlet   = DirichletBC(
                self.V.sub(0), self.outlet, outlet_bdry
            )

        self.bcs = [bc_inlet, bc_outlet]

        # # BUILD SOLVER
        J       = derivative(self.vp, self.U) # ? Jacobian
        problem = NonlinearVariationalProblem(self.vp, self.U, self.bcs, J=J)
        self.solver = NonlinearVariationalSolver(problem)
        self.solver.parameters.update(solver_params)


    def crank_nicholson_scheme(self, dx, A, u, An, un, psi_a, psi_u):
        """
        Variational form using a Crank-Nicholson time scheme.
        """
        
        # Area Equation
        AF = A*psi_a*dx     \
            - An*psi_a*dx   \
            + self.dt * (
                self.theta       * ( grad(A *u )[0]*psi_a*dx )
                + (1-self.theta) * ( grad(An*un)[0]*psi_a*dx )
            )

        # Momentum Equation
        c2 = self.beta0/2/self.rho/self.A0 
        # ? not full c2, missing 'sqrt(A)/A' but it is included later

        # Momentum correction coefficient [-]
        alpha = (self.gamma+2)*(self.gamma+1)**-1

        uF = u*psi_u*dx     \
            - un*psi_u*dx   \
            + self.dt * (
                #? Momentum Advection
                self.theta       * alpha * u  * grad(u)[0]  * psi_u * dx
                + (1-self.theta) * alpha * un * grad(un)[0] * psi_u * dx
                #? Pressure Forcing
                + self.theta     * ( ( c2 / pow(A +DOLFIN_EPS,0.5) ) * grad(A )[0] * psi_u * dx )
                + (1-self.theta) * ( ( c2 / pow(An+DOLFIN_EPS,0.5) ) * grad(An)[0] * psi_u * dx )
                #? Viscous Dissipation
                + self.theta     * ( self.Kr * u  / (A +DOLFIN_EPS) ) * psi_u * dx
                + (1-self.theta) * ( self.Kr * un / (An+DOLFIN_EPS) ) * psi_u * dx
                #? Reference Area Pressure Forcing
                + self.theta     * (
                      ( self.beta0 / self.rho )
                    * ( ( 1 / ( 2 * pow(self.A0,1.5) ) ) - ( pow(A +DOLFIN_EPS,0.5) / pow(self.A0,2) ) )
                    * self.dA0 
                ) * psi_u * dx
                + (1-self.theta) * ( 
                      ( self.beta0 / self.rho )
                    * ( ( 1 / ( 2 * pow(self.A0,1.5) ) ) - ( pow(An+DOLFIN_EPS,0.5) / pow(self.A0,2) ) )
                    * self.dA0 
                ) * psi_u * dx
                #? Vessel Material Pressure Forcing
                + self.theta     * ( 
                      ( 1 / self.rho / self.A0 ) 
                    * ( pow(A +DOLFIN_EPS,0.5) - pow(self.A0,0.5) ) 
                    * self.dbeta0 
                ) * psi_u * dx
                + (1-self.theta) * ( 
                      ( 1 / self.rho / self.A0 ) 
                    * ( pow(An+DOLFIN_EPS,0.5) - pow(self.A0,0.5) ) 
                    * self.dbeta0 
                ) * psi_u * dx
            )
        
        return AF + uF
    

    def solve(self):
        """
        Solve the non-linear system built in define_variational_problem().
        """

        self.solver.solve()
    

    def update_solution(self):
        """
        Update the previous solution.
        """

        self.Un.assign(self.U)


    def update_pressure(self):
        """
        Update the pressure based on the previous solution.
        """
        self.pn.assign(
            Expression(
                'p0 + ( beta / Ai ) * ( sqrt(A) - sqrt(Ai) )',
                degree=2, p0=self.pd, beta=self.beta, A=self.Un.sub(0), Ai=self.A0
            )
        )


    def evaluate_pressure_at_x(self, x):
        '''
        Evaluates the pressure at a specified location, x=[0,1], using the tube law.
        '''

        return self.pd + ( self.beta0p(x,) / self.A0p(x,) ) * (sqrt(self.Un(x,)[0]) - sqrt(self.A0p(x,)))


    def probe_spatial_solutions(self, x):
        '''
        Evaluates vessel solutions at the specified location, x.
        '''

        An, Un = self.Un(x,)

        assert An > 0, 'negative area is no real'

        return array( (
                An,
                Un
            ),
            dtype=sol_dtype
        )


    def probe_spatial_properties(self, x):
        '''
        Evaluates vessel properties at the specified location, x.
        '''

        return array( (
                self.A0(x,),
                self.u0,        # ! u0 may not be constant
                self.beta0(x,),
                self.rho
            ),
            dtype=prop_dtype
        )