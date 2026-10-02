"""
auth: L. J. Barratt
Created on 15/06/26
"""

from scipy.linalg import lu_factor, lu_solve
from numpy        import array, linalg

#%% CLASS

class MathUtilities:

    @staticmethod
    def netwon_lu_solver(fun, jac, x0, args=(), tol=1e-8, max_iter=50):
        '''
        LU solver with Newton-Raphson iterator loop and line-search.
        '''

        x = array(x0, dtype=float)
        area_idx = [0, 2, 4]

        # # NEWTON-RAPHSON LOOP
        for i in range(max_iter):

            # evaluate residual and Jacobian
            F = array(fun(x, *args), dtype=float)
            J = array(jac(x, *args), dtype=float)
            
            # LU factorization and solve J * delta = -F
            lu_piv = lu_factor(J)
            delta  = lu_solve(lu_piv, -F)
            
            # backtracking line search
            alpha    = 1.0
            F_norm   = linalg.norm(F, ord=2)
            max_back = 30
            

            # # LINE SEARCH 
            for _ in range(max_back):

                x_new = x + alpha * delta
                
                # ensure areas remain positive
                if all(x_new[k] > 0 for k in area_idx):
                    F_new = array(fun(x_new, *args), dtype=float)
                    # accept step if residual decreased
                    if linalg.norm(F_new, ord=2) < F_norm:
                        break
                
                # reduce step size and retry
                alpha *= 0.5

            # ? line search failed — take smallest safe step
            else:
                alpha = 0.5**max_back
            

            # # UPDATE SOLUTION
            x = x + alpha * delta
            
            # ? clamp on area: must be positive
            for k in area_idx:
                x[k] = max(x[k], 1e-12)
            
            # # CHECK CONVERGENCE
            if linalg.norm(delta * alpha, ord=2) < tol:
                return x
        

        raise ValueError(f'Newton-Raphson did not converge after {max_iter} iterations')