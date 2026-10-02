"""
auth: L. J. Barratt
date: 01/09/26
"""

from scipy.linalg import lu_factor, lu_solve
from numpy import sqrt, array, linalg, abs, zeros, all, isfinite, maximum, inf
from dolfin import FunctionSpace, interpolate


# # ############################################################################
# # ARTERIAL ROOT CONDITION
# # ############################################################################

def inflow_rate_adaptive(
        artery, Qin, 
        tol=1e-12, max_iter=1000, 
        alpha=1.0, min_alpha=0.1, alpha_decay=0.7
    ):
    """
    Approach to employ a flow condition based on Alastruey, et al. (2012).

    Compute the unique state (As, Us) where Qin=As*Us and Wb(As,Us)=Wb(Ar,Ur). Automatically adjusts damping factor if convergence fails.
    """

    # # CURRENT CONDITIONS
    Ar, Ur = artery.Un(0,)
    As = Ar                         # initial guess for area
    current_alpha = alpha           # starting damping coefficient

    # # NEWTON STEPPING WITH LINE SEARCH
    for _ in range(10):  # Try reducing alpha up to 10 times if necessary

        As = Ar  # Reset initial guess for each attempt
        for idx in range(max_iter):
            tmp = pow(artery.beta / 2 / artery.rho / artery.A0i, 0.5)
            W2 = Ur - 4 * pow(Ar, 0.25) * tmp
            cAs = pow(As, 0.25) * tmp
            dcdA = 0.25 * pow(As, -0.75) * tmp

            R = Qin - As * W2 - 4 * As * cAs
            dRdA = W2 + 4 * (cAs + As * dcdA)

            if abs(R) < tol:
                #!print(f'Proximal Conditons: Newton-Raphson method converged in {idx} steps.')
                Us = W2 + 4 * cAs
                return As, Us, Ur

            As += current_alpha * (R / dRdA)

        # If we reach here, Newton-Raphson did not converge; reduce damping
        current_alpha *= alpha_decay
        if current_alpha < min_alpha:
            break  # Stop if damping becomes too small

    raise RuntimeError(f'Proximal Conditions: Newton-Raphson method did not converge after adjusting damping. Last alpha={current_alpha:.2f}')


# # ############################################################################
# # LU SOLVER IN NEWTON-RAPHSON ITERATIVE LOOP WITH LINE-SEARCH
# # ############################################################################

def nrluls(junction, jacobian, x0, args=(), tol=1e-9, max_iter=50):

    x = array(x0, dtype=float)

    if not all(isfinite(x)):
        raise ValueError(f'Non-finite initial guess: {x}')
    
    area_idx = [0, 2, 4]

    for it in range(max_iter):

        F = array(junction(x, *args), dtype=float)
        J = array(jacobian(x, *args), dtype=float)

        if not (all(isfinite(F)) and all(isfinite(J))):
            raise ValueError(
                f'Non-finite residual/Jacobian at iter {it}, x={x}'
            )

        # row-equilibrate so no single equation dominates the merit function
        r = 1.0 / maximum(abs(J).max(axis=1), 1e-300)
        Fs, Js = F * r, J * r[:, None]

        Fnorm = linalg.norm(Fs, inf)
        if Fnorm < tol:
            return x

        delta = lu_solve(lu_factor(Js), -Fs)

        # backtracking line search (Armijo)
        alpha = 1.0
        for _ in range(30):
            
            x_new = x + alpha * delta
            
            if all(x_new[k] > 0 for k in area_idx):
                Fn = array(junction(x_new, *args), dtype=float) * r
                
                if all(isfinite(Fn)) and \
                   linalg.norm(Fn) <= (1 - 1e-4 * alpha) * linalg.norm(Fs):
                    
                    break

                
            alpha *= 0.5

        else:
            raise ValueError(
                f'Line search failed at iter {it}; |F|={Fnorm:.3e}, x={x}'
            )

        
        x = x_new

    raise ValueError(
        f'No convergence in {max_iter} iters; |F|={Fnorm:.3e}, x={x}'
    )


# # ############################################################################
# # JUNCTION - SPLIT
# # ############################################################################

def splitting(a1, a2, a3, ids=None, _cache={}):
    """
    Junction solver for a splitting arterial junction:
    Parent artery (a1) → Daughter arteries (a2, a3).
    Solves for the areas and velocities (A*, U*) at the junction.
    """

    # * EQUATIONS
    def P_e(A, A0, b, P_ext=0.0):
        """Tube law (pressure–area relationship)."""
        return P_ext + (b / A0) * (sqrt(A) - sqrt(A0))

    def wave_speed(A, A0, b, rho):
        """Wave speed at area A."""
        return sqrt(b / (2.0 * rho * A0)) * A**0.25

    def wave_speed_ref(A0, b, rho):
        """Wave speed at reference configuration."""
        return sqrt(b / (2.0 * rho * A0)) * A0**0.25

    def W_f(A, U, A0, U0, b, rho):
        """Forward Riemann invariant."""
        c  = wave_speed(A, A0, b, rho)
        c0 = wave_speed_ref(A0, b, rho)
        return (U - U0) + 4.0 * (c - c0)

    def W_b(A, U, A0, U0, b, rho):
        """Backward Riemann invariant."""
        c  = wave_speed(A, A0, b, rho)
        c0 = wave_speed_ref(A0, b, rho)
        return (U - U0) - 4.0 * (c - c0)


    # * ANALYTICAL JACOBIAN
    def jacobian(
        vars,
        An_a, Un_a, An_b, Un_b, An_c, Un_c,
        A0a, U0a, A0b, U0b, A0c, U0c,
        ba, bb, bc,
        rhoa, rhob, rhoc
    ):
        
        As_a, Us_a, As_b, Us_b, As_c, Us_c = vars

        # Pressure derivatives
        dPe_a = ba / (2*A0a*sqrt(As_a))
        dPe_b = bb / (2*A0b*sqrt(As_b))
        dPe_c = bc / (2*A0c*sqrt(As_c))

        # Wave speeds
        c_a = sqrt(ba / (2*rhoa*A0a)) * As_a**0.25
        c_b = sqrt(bb / (2*rhob*A0b)) * As_b**0.25
        c_c = sqrt(bc / (2*rhoc*A0c)) * As_c**0.25

        dc_a = c_a / (4*As_a)
        dc_b = c_b / (4*As_b)
        dc_c = c_c / (4*As_c)

        J = zeros((6,6))

        # Mass
        J[0,0] = Us_a
        J[0,1] = As_a
        J[0,2] = -Us_b
        J[0,3] = -As_b
        J[0,4] = -Us_c
        J[0,5] = -As_c

        # Bernoulli f2
        J[1,0] = dPe_a
        J[1,1] = rhoa*Us_a
        J[1,2] = -dPe_b
        J[1,3] = -rhob*Us_b

        # Bernoulli f3
        J[2,0] = dPe_a
        J[2,1] = rhoa*Us_a
        J[2,4] = -dPe_c
        J[2,5] = -rhoc*Us_c

        # Characteristic f4 (parent Wf)
        # J[3,0] = dc_a / (dc_a / (c_a/As_a))
        J[3,0] = 4*dc_a
        J[3,1] = 1

        # Characteristic f5 (daughter b Wb)
        J[4,2] = -4*dc_b
        J[4,3] = 1

        # Characteristic f6 (daughter c Wb)
        J[5,4] = -4*dc_c
        J[5,5] = 1

        return J

    # * RESIDUAL FUNCTION
    def junction(
        vars, 
        An_a, Un_a, An_b, Un_b, An_c, Un_c, 
        A0a, U0a, A0b, U0b, A0c, U0c, 
        ba, bb, bc, 
        rhoa, rhob, rhoc
    ):
        As_a, Us_a, As_b, Us_b, As_c, Us_c = vars

        # Mass conservation
        f1 = As_a * Us_a - As_b * Us_b - As_c * Us_c

        # Bernoulli energy between parent and each daughter
        f2 = (P_e(As_a, A0a, ba) + 0.5 * rhoa * Us_a**2) - \
                (P_e(As_b, A0b, bb) + 0.5 * rhob * Us_b**2)
        f3 = (P_e(As_a, A0a, ba) + 0.5 * rhoa * Us_a**2) - \
                (P_e(As_c, A0c, bc) + 0.5 * rhoc * Us_c**2)

        # Characteristic matching
        f4 = W_f(As_a, Us_a, A0a, U0a, ba, rhoa) - W_f(An_a, Un_a, A0a, U0a, ba, rhoa)
        f5 = W_b(As_b, Us_b, A0b, U0b, bb, rhob) - W_b(An_b, Un_b, A0b, U0b, bb, rhob)
        f6 = W_b(As_c, Us_c, A0c, U0c, bc, rhoc) - W_b(An_c, Un_c, A0c, U0c, bc, rhoc)

        return [f1, f2, f3, f4, f5, f6]
    
    
    # * REFERENCE AND PREVIOUS STEP STATES
    (s1, p1), (s2, p2), (s3, p3) = a1, a2, a3

    A0a, U0a, betaa, rhoa = p1['A0'], p1['u0'], p1['beta'], p1['rho']
    An_a, Un_a            = s1['An'], s1['Un']

    A0b, U0b, betab, rhob = p2['A0'], p2['u0'], p2['beta'], p2['rho']
    An_b, Un_b            = s2['An'], s2['Un']

    A0c, U0c, betac, rhoc = p3['A0'], p3['u0'], p3['beta'], p3['rho']
    An_c, Un_c            = s3['An'], s3['Un']

    # ! the cache key stores the solutions of the pevious solve
    # ! use this of the current values to see if things have changes
    # ! if not, re-apply the old solution and skip the solve
    # ! if yes, use the 

    # build cache key from artery reference areas (unique per junction)
    cache_key = ids if ids is not None else (a1['A0'], a2['A0'], a3['A0'])
    x0_cold = [An_a, Un_a, An_b, Un_b, An_c, Un_c]
    x0 = _cache.get(cache_key, x0_cold)

    # * SOLVE
    args = (
        An_a, Un_a, An_b, Un_b, An_c, Un_c,
        A0a, U0a, A0b, U0b, A0c, U0c,
        betaa, betab, betac,
        rhoa, rhob, rhoc,
    )
    assert all(isfinite(args)), "NaN/inf in junction inputs"

    # ? try first with cached values
    try:
        solution = nrluls( junction, jacobian, x0, args=args )
    except ValueError:
        if x0 is x0_cold:
            raise
        
        solution = nrluls( junction, jacobian, x0_cold, args=args )


    As_a, Us_a, As_b, Us_b, As_c, Us_c = solution
    _cache[cache_key] = list(solution)

    return ( [As_a, Us_a], [As_b, Us_b], [As_c, Us_c] )


# # ############################################################################
# # STABILITY CHECK FOR ARTERY MODEL
# # ############################################################################
def _vertex_coefficients(artery):
    """beta(x), A0(x) at mesh vertices, computed once and cached."""

    if not hasattr(artery, '_cfl_cache'):
        V1 = FunctionSpace(artery.mesh, 'CG', 1)
        beta_v = interpolate(artery.beta0, V1).compute_vertex_values(artery.mesh)
        A0_v   = interpolate(artery.A0,    V1).compute_vertex_values(artery.mesh)
        artery._cfl_cache = (beta_v, A0_v)

    return artery._cfl_cache


def check_stability(idx, artery, dt, disp=False, u_limit=3.0):
    '''
    Function to check the stability of a coupled 
    hyperbolic system via the CFL condition:
        CFL = dt * max(| u + c |) <= 0.5 * dx.

    Includes sanity checks on A and U.
    '''

    area, vel = artery.Un.split()
    A = area.compute_vertex_values(artery.mesh)
    U = vel.compute_vertex_values(artery.mesh)
    n = len(A) - 1

    if not (all(isfinite(A)) and all(isfinite(U))):
        raise RuntimeError(f'Artery {idx}: non-finite A or U')
    if A.min() <= 0.0:
        raise RuntimeError(f'Artery {idx}: non-positive area {A.min():.3e} at vertex {A.argmin()}/{n}')

    beta_v, A0_v = _vertex_coefficients(artery)
    c     = sqrt(beta_v / (2.0 * artery.rho * A0_v)) * A**0.25
    speed = abs(U) + c
    k     = speed.argmax()
    cfl   = dt * speed[k] / artery.dx

    ku, umax = abs(U).argmax(), abs(U).max()
    
    if disp:
        print(
            f'Artery {idx}: CFL={cfl:.3f} (vertex {k}/{n}), '
            f'|U|max={umax:.3f} (vertex {ku}/{n}), Amin={A.min():.3e}'
        )
        
    if cfl >= 0.5:
        raise RuntimeError(
            f'Artery {idx} unstable: CFL={cfl:.4f} at vertex {k}/{n}'
        )
    
    if umax > u_limit:
        raise RuntimeError(
            f'Artery {idx}: |U|={umax:.2f} m/s at vertex {ku}/{n} exceeds {u_limit}'
        )