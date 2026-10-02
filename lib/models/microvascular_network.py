"""
auth: L. J. Barratt
date: 03/11/25

Class housing methods to solve Hagen-Poiseuille flow in an asymmetric fractal.
"""

#%% PACKAGES

#* EXTERNAL LIBRARIES
from numpy       import sqrt, pi, arange, exp, heaviside, where,     \
                        array, zeros, errstate, nan, mean, fromiter, \
                        full_like
from typing      import List, Dict
from dataclasses import dataclass
import time

#* INTERNAL LIBRARIES


#%% DATA CLASSES
@dataclass
class Vessel:
    i:  int          # Parent id
    j:  int          # Vessel/Child id
    r0: float        # Initial radius
    r:  float        # Current radius
    L:  float        # Length
    V:  float        # Volume
    m:  float        # Apparent viscosity
    c:  int          # Vessel type (0, artery; 1, arteriole; 2, capillary)
    g:  float        # Resistance
    b:  float        # Material coefficient
    C0: float        # Initial compliance
    C:  float        # Current compliance
    q:  float        # Flow
    t:  float        # Shear stress
    T:  float        # Wall tension


#%% CLASS
class MicrovascularNetwork:

    def __init__(self, tree, fluid, comm=None):

        # * CONFIGURE MPI
        self.comm = comm

        # ? Parallel proccessing: PETSc solver
        if self.comm is not None:

            from lib.etc.petsc_network_backend \
                    import PETScNetworkSolver

            Backend = PETScNetworkSolver


        # ? Serial proccessing: PARDISO solver
        else:

            from lib.etc.pardiso_network_backend \
                    import PARDISONetworkSolver

            Backend = PARDISONetworkSolver

        
        self.solver = Backend(self)

        # # ROOT AND TERMINAL GEOMETRY
        self.r0    = tree['root_radius'][0]   # Root radius 
        self.rcut  = tree['cutoff_radius'][0] # Cutoff radius

        # # TREE GEOMETRY PARAMETERS
        self.lmbda = 1.0
        # length to radius ratio
        self.Lrmax = tree['length_to_radius'][0]       
        # theshold for Lr [m]
        self.rthr  = tree['threshold_radius'][0] 

        #? parameters varied by vessel type:
        self.gamma = tree['symmetry']  # Symmetry coefficient 
        self.zeta  = tree['shrinkage'] # Shrinkage coefficient
        self.branch_params = {
            "artery":    (self.gamma[0], self.zeta[0]),
            "arteriole": (self.gamma[1], self.zeta[1]),
            "capillary": (self.gamma[2], self.zeta[2]),
        }

        # # FLUID PROPERTIES
        self.C         = pi / 8  # Constant vessel resistance
        self.mu        = fluid['mu'][0]  # Viscosity [Pa s]
        
        # fluid viscosity type
        self.newtonian = (fluid.get('newtonian'))[0].lower() \
                == 'true'

        if self.newtonian is False:
            # Relative Haematocrit discharge
            self.hctrel = fluid['relative_hd'][0]


        # # BOUNDARY CONDITIONS
        inlet_type = tree.get('inlet_type')[0]
        self.root     = 0.0
        self.terminal = 0.0

        if inlet_type.lower() == 'flow':
            self.inflow_type = 1

        elif inlet_type.lower() == 'pressure':
            self.inflow_type = 0
        
        else:
            raise ValueError(
                f'{inlet_type} is not valid inflow type. Options: flow, pressure.'
            )

        self.lim = tree['class_limits']
        if len(self.lim) > 2:
            raise ValueError('eat the rich')


        # # BUILD NUMERICAL SYSTEM
        self.generate_fractal()
        self.classify_vessels()
        self.solver.initialise_system()


    def generate_fractal(self):
        '''
        Script to generate an asymmetric fractal to be used in 
        modelling Hagen-Poiseuille flow in the microvasculature. 
        The method used is based on the one used by Olufsen, et al. 
        (1998,1999). 

        In parallel, all ranks create all vessels.

        Process:
        --------
        The radii of the daughers of a vessel are scaled by as:
        
            r1 = alpha * r0
            r2 = beta  * r0.

        The fractal terminates when the radius becomes smaller than 
        the cutoff radius.
        '''

        # List of vessels with parameters
        self.vessels: List[Vessel] = []
        
        # List of all terminal vessel ids
        self.terminals = []

        # List of all vessels and their fractal depth
        self.node_levels: Dict[int, int] = {0: 0} 
        
        # Node index
        node_idx = 1

        # Stack for vessels to be generated
        # This holds the parent id, parent radius, and level 
        vessel_stack = [(0, self.r0, 0)]

        # Generate tree
        while vessel_stack:

            #? pop off the most recent vessel addition
            pid, rp, lvl = vessel_stack.pop()

            #? identify vessel classficiation
            vtype = self.get_vessel_type(rp)

            #? select appropriate gamma and zeta values
            gamma, zeta = list( self.branch_params.values() )[vtype]
            
            #? compute alpha and beta
            alpha = ( 1 + gamma ** ( zeta / 2 ) ) ** ( -1 / zeta )
            beta  = alpha * sqrt(gamma)

            #? compute radii of daughter vessels
            r1 = alpha * rp
            r2 = beta  * rp

            #? check if daughters are terminals
            if ( r1 < self.rcut ) and ( r2 < self.rcut ):
                self.terminals.append(pid)
                continue    # skip the later loop

            #? loop through daughters 
            for rd in (r1, r2):

                #? end the branch if the vessel radius is below cutoff
                if rd < self.rcut:
                    continue

                #? child id / current vessel id
                cid = node_idx
                self.node_levels[cid] = lvl+1
                
                #? identify vessel classficiation
                vcl = self.get_vessel_type(rp)
                    # * 0, artery; 1, arteriole; 2, capillary

                #? child length
                Ld = self.compute_length_to_radius_ratio(rd) * rd

                #? child volume
                Vd = pi * rd**2 * Ld

                #? child viscosity
                mud = self.compute_apparent_viscosity(rd)

                #? child resistance
                gd  = self.C * ( rd**4 ) / ( Ld * mud )
                
                #? child stiffness
                Eh = self.compute_vessel_stiffness(rd)

                #? material coefficient
                nu   = 0.5 # poisson's ratio, incompressible
                beta = Eh * sqrt(pi) / (1.0 - nu**2)

                # compliance, dV/dp 
                # uses tube law: p = p_0 + ( beta/A_0 )( sqrt(A) - sqrt(A_0) ) )
                A0 = pi * rd**2
                C  = 2.0 * (A0 ** 1.5) * Ld / beta

                #? update node index
                node_idx += 1

                #? add new vessel to list
                self.vessels.append( 
                    Vessel(
                        pid, cid,           # indices
                        rd, rd, Ld, Vd,     # geometry
                        mud,                # viscosity
                        vcl,                # vessel class
                        gd,                 # resistance
                        beta,               # material coefficient
                        C, C,               # compliance
                        0, 0, 0             # flow / shear / tension
                    ) 
                )

                #? add vessel to stack to find its childen
                vessel_stack.append( (cid, rd, lvl+1) )

        
        # Number of nodes
        self.N = node_idx


    def get_vessel_type(self, radius):

        if radius*1e6 > self.lim[1]:
            return 0    # ? artery
        
        elif radius*1e6 > self.lim[0]:
            return 1    # ? arteriole
        
        else:
            return 2    # ? capillary


    def compute_length_to_radius_ratio(self, radius):
        '''
        Computes a radius-dependent length-to-radius ratio. The ratio 
        is constant above a specified threshold; afterwhich, it 
        decreases linearly with the radius.
        '''

        if radius >= self.rthr:
            return self.Lrmax

        else:
            return ( self.Lrmax / self.rthr ) * radius
        

    def compute_apparent_viscosity(self, radius):
        '''
        Computes a radius-dependent viscosity.
        '''

        if self.newtonian:
            return full_like(radius, self.mu, dtype=float)
            # return self.mu
        
        # Vessel diameter (in micro meters)
        D = 2*radius*1e6

        # Compute relative apparent viscosity
        alpha = 4 / ( 1 + exp( -0.593 * ( D-6.74 ) ) )
        murelapp = 1 + (
            ( exp(self.hctrel*alpha)-1 ) / ( exp(0.45*alpha)-1 ) ) \
          * ( 420*exp( -0.1*D ) + 3 - 3.45*exp( -0.035*D )
        )

        # Return apparent viscosity
        return self.mu * murelapp
    

    def compute_vessel_stiffness(self, radius):
        '''
        Computes vessel stiffness using the empirical relation from 
        Olufsen, et al. (1999).
        Since the exponential fit becomes singular for r<500μm, 
        vessels below this threshold have the same elastic modulus 
        corresponding to r=500μm as predicted by Olufsen's relation
        (Perdikaris, et al. 2015). 

        Returns
        -------
        Eh : float
            Young's Modulus multiplied by the vessel thickness.
        '''

        k1 = 2e7
        k2 = -22.53
        k3 = 8.65e5

        # if radius < 5e-4: radius = 5e-4
        radius = 5e-4 if radius < 5e-4 else radius
        
        return ( k1 * exp(k2 * radius) + k3 ) * radius
        

    def classify_vessels(self):
        '''
        Classify vessels into groups (e.g., arterioles, capillaries)
        based on their baseline radius.
        '''

        # Convert to micrometers for readability
        radii_um = array([v.r * 1e6 for v in self.vessels])

        #! keep for now
        #! steele uses: 250>art | 250>atl<50 | cap>50
        self.vessel_groups = {
            'arteries': where((radii_um >= self.lim[1]))[0],
            'arterioles': where(
                (radii_um >= self.lim[0]) & (radii_um < self.lim[1])
            )[0],
            'capillaries': where((radii_um < self.lim[0]))[0]
        }

        # ! print output
        # total = 0
        # for k, idx in self.vessel_groups.items():
        #     self.print(f"\t{k}\t: {len(idx)}")
        #     total += len(idx)


        # self.print(f'\ttotal\t\t: {total}')

        # assert total == len(self.vessels),  \
        #     'Vessels are present that do not adhere to ' \
        #     'the classifications of vessel groups.'


    def update_vessel_list(self, lmbda, groups=None):
        '''
        Updates the radius of all (or a subgroup of) vessels in the
        network based on the given dilation factor lmbda and updates the local value (self.lmbda).

        Following this, the viscosity, conductance, and volume of
        the group of vessels is updated.

        If group is None, all vessels are updated. Raises a ValueError
        if the specified group is not defined in the network. 

        In parallel, all ranks update all vessels. # TODO improve this

        Parameters
        ----------
        groups : list[str] or None
            Group names to average over. If None, averages over the
            whole network.
        '''

        #* check group
        if groups is None:
            v_indices = range(len(self.vessels)) # indices for all vessels

        else:
            invalid_groups = [ 
                g 
                for g in groups 
                if g not in self.vessel_groups 
            ]
            if invalid_groups:
                raise ValueError('Group(s) {invalid_groups} not defined.')

            # indices for chosen vessels
            v_indices = [
                i
                for g in groups
                for i in self.vessel_groups.get(g, [])
            ]

        # # update parameters
        # ! this is repeated everytime, but the reference values do not change
        vessels = self.vessels
        r0 = fromiter((vessels[i].r0 for i in v_indices), dtype=float)
        L  = fromiter((vessels[i].L  for i in v_indices), dtype=float)
        C0 = fromiter((vessels[i].C0 for i in v_indices), dtype=float)

        # update radius based on dilation changes
        self.lmbda = lmbda
        r = r0 * self.lmbda

        # new viscosity
        m = self.compute_apparent_viscosity(r)

        # new resistance
        g = pi * r**4 / (8 * m * L)

        # new volume
        V = pi * r**2 * L

        # new compliance
        C = C0 * self.lmbda
        
        # # update list of vessels
        for idx, ri, mi, gi, Vi, Ci in zip(v_indices, r, m, g, V, C):
            v = vessels[idx]
            v.r = ri
            v.m = mi
            v.g = gi
            v.V = Vi
            v.C = Ci
        

    def solve(self):
        self.solver.solve()

    
    def compute_vessel_parameters(self):
        '''
        Computes vessel parameters after a solution has been computed.

        Should be called whenever a new solution has been computed.

        In parallel, this method will only operate on Rank 0. 
        Therefore, the flow and shear in self.vessels is only kept up 
        to date on Rank 0.
        '''

        # give the pressure array to Rank 0
        p = self.solver.get_pressure_array()

        # other ranks will returns
        if ( self.comm is not None ) and self.comm.Get_rank() != 0:
            return

        for ves in self.vessels:

            # ? pressure drop across vessel
            dp = p[ves.i] - p[ves.j]

            # ? flow through vessel
            ves.q = ves.g * dp

            # ? shear stress
            ves.t = ( ves.r * dp ) / ( 2 * ves.L )

            # ? wall tension from average pressure across vessel
            ves.T = ( ( p[ves.i] + p[ves.j] ) / 2 )  * ves.r

            # if ves.i < 50:
            #     print(f'test: \t {ves.i} \t {p[ves.i]:.6f} \t {ves.r:.6f} \t {ves.T:.6f}')
        

    def compute_level_averages(self):
        '''
        Computes average properties at each fractal level.

        Requires the vessel parameters be up to date. Use compute_vessel_parameters()
        any time a new solution for p is computed.

        In parallel, this method will only operate on Rank 0 and returns
        None(s) on all other ranks.
        
        Returns
        -------
        nlvls_arr : ndarray
            Integer node-level indices.
        vlvls_arr : ndarray
            Vessel-level indices offset by 0.5.
        pavg : ndarray
            Mean pressure at each nodal level in Pa.
        qavg : ndarray
            Mean flow rate at each vessel level in m³/s.
        tavg : ndarray
            Mean wall shear stress at each vessel level in Pa.
        qtot : ndarray
            Total flow rate at each vessel level in m³/s.
        '''

        # give the pressure array to Rank 0
        p = self.solver.get_pressure_array()

        # other ranks will return None(s)
        if ( self.comm is not None ) and self.comm.Get_rank() != 0:
            return None, None, None, None, None, None
        

        # Number of nodal/vessel levels
        vlvls = max(self.node_levels.values())
        nlvls = vlvls + 1

        # array init
        p_sum = zeros(nlvls)
        p_cnt = zeros(nlvls)

        q_sum = zeros(vlvls)
        q_cnt = zeros(vlvls)

        t_sum = zeros(vlvls)
        t_cnt = zeros(vlvls)

        # sum of pressures and nodal count at each level
        for n, lvl in self.node_levels.items():
            p_sum[lvl] += p[n]
            p_cnt[lvl] += 1


        # sum of flow/shear and vessel count at each level
        for ves in self.vessels:
            lvl = self.node_levels[ves.i]

            q_sum[lvl] += ves.q
            q_cnt[lvl] += 1

            t_sum[lvl] += ves.t
            t_cnt[lvl] += 1


        # level averages (ignoring divisions by zero)
        with errstate(invalid='ignore'):
            pavg = p_sum / p_cnt
            qavg = q_sum / q_cnt
            tavg = t_sum / t_cnt


        # total flow at each level
        qtot = q_sum.copy()

        # replace divisions by zero with nan
        pavg[p_cnt == 0] = nan
        qavg[q_cnt == 0] = nan
        tavg[t_cnt == 0] = nan
        qtot[q_cnt == 0] = nan

        nlvls_arr = arange(nlvls)
        vlvls_arr = arange(vlvls) + 0.5

        return nlvls_arr, vlvls_arr, pavg, qavg, tavg, qtot


    def compute_tree_volume(self):
        '''
        Compute the total and per-group vascular volume of the fractal tree.

        In parallel, this method will only operate on Rank 0 and returns
        (None, None) on all other ranks.

        Returns
        -------
        total_volume : float or None
            Total vascular volume in m³, or None if not Rank 0.
        group_volumes : dict[str, float] or None
            Per-group volumes in m³ keyed by group name, or None if not Rank 0.
        '''

        # other ranks will return None(s)
        if ( self.comm is not None ) and self.comm.Get_rank() != 0:
            return None, None
            

        # build reverse lookup: vessel index -> group name
        # cache it so repeated calls don't recompute
        if not hasattr(self, '_vessel_to_group'):
            vessel_groups = getattr(self, 'vessel_groups', {})
            self._vessel_to_group = {}
            for group_name, vids in vessel_groups.items():
                for vid in vids:
                    self._vessel_to_group[vid] = group_name

        # pass over all vessels, add volume to total and group
        group_volumes = {
            name: 0.0 for name in getattr(self, 'vessel_groups', {})
        }
        total_volume  = 0.0

        for i, ves in enumerate(self.vessels):
            v = ves.V
            total_volume += v
            group_name = self._vessel_to_group.get(i)
            if group_name is not None:
                group_volumes[group_name] += v


        return total_volume, group_volumes


    def compute_shear_average(self, groups=None):
        '''
        Compute the mean wall shear stress across the network or a subset.

        Requires the vessel parameters be up to date. Use compute_vessel_parameters()
        any time a new solution for p is computed.

        In parallel, this method will only operate on Rank 0 and returns
        None on all other ranks.

        Parameters
        ----------
        groups : list[str] or None
            Group names to average over. If None, averages over the whole network.

        Returns
        -------
        float or None
            Mean wall shear stress in Pa, or None if not Rank 0 or no vessels found.
        '''

        # other ranks will return None
        if ( self.comm is not None ) and self.comm.Get_rank() != 0:
            return None


        if groups is None:
            shear_values = [v.t for v in self.vessels]

        else:
            vessel_indices = [
                i 
                for g in groups 
                for i in self.vessel_groups.get(g, [])
            ]
            shear_values = [self.vessels[i].t for i in vessel_indices]


        if not shear_values:
            raise ValueError('problem with shear average...')


        return mean(shear_values)


    def compute_tension_average(self, groups=None):
        '''
        Compute the mean wall tension across the network or a subset.

        Requires the vessel parameters be up to date. Use compute_vessel_parameters()
        any time a new solution for p is computed.

        In parallel, this method will only operate on Rank 0 and returns
        None on all other ranks.

        Parameters
        ----------
        groups : list[str] or None
            Group names to average over. If None, averages over the whole network.

        Returns
        -------
        float or None
            Mean wall shear tension in Pa.m, or None if not Rank 0.
        '''

        # other ranks will return None
        if ( self.comm is not None ) and self.comm.Get_rank() != 0:
            return None


        if groups is None:
            tension = [v.T for v in self.vessels]

        else:
            vessel_indices = [
                i 
                for g in groups 
                for i in self.vessel_groups.get(g, [])
            ]
            tension = [self.vessels[i].T for i in vessel_indices]


        if not tension:
            raise ValueError('problem with tension average...')


        return mean(tension)


    def compute_tree_resistance(self):
        '''
        Computes total effective resistance across the network.

        Requires the vessel parameters be up to date. Use compute_vessel_parameters()
        any time a new solution for p is computed.

        In parallel, this method will only operate on Rank 0 and returns
        None on all other ranks.

        Returns
        -------
        R_total : float
            Effective total resistance (Pa·s/m³)
        '''

        # give the pressure array to Rank 0.
        p = self.solver.get_pressure_array()

        # other ranks will return None
        if ( self.comm is not None ) and self.comm.Get_rank() != 0:
            return None
        
        
        root_idx = 0

        # get root and terminal pressure
        p_in  = p[root_idx]
        p_out = self.terminal
        dp    = p_in - p_out

        # get root flow
        q_in = self.vessels[root_idx].q

        # print(p_in, p_out, dp, q_in)

        # check
        if abs(q_in) < 1e-20:
            raise ValueError('Inlet flow is zero; cannot compute total resistance.')
        

        # compute and return total resistance
        return dp / q_in


    def compute_tree_compliance(self):
        """
        Compute total compliance of a vascular tree using a tube law 
        linearised around A0.

        Requires the vessel parameters be up to date. Use compute_vessel_parameters()
        any time a new solution for p is computed.

        In parallel, this method will only operate on Rank 0 and returns
        None on all other ranks.

        Returns
        -------
        ctot : float
            Total tree compliance
        """

        # other ranks will return None
        if ( self.comm is not None ) and self.comm.Get_rank() != 0:
            return None

        Ci = fromiter((v.C for v in self.vessels), dtype=float)

        return sum(Ci)


    def compute_root_flow(self):
        return self.vessels[0].q + self.vessels[1].q
    
