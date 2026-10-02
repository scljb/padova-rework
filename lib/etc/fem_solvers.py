#%% LIBRARIES
from fenics import set_log_level, LogLevel


#%% FEniCS SETTINGS
set_log_level(LogLevel.ERROR)
# set_log_level(LogLevel.PROGRESS) 
# set_log_level(LogLevel.DEBUG)

"""
check out AMG solver

"""


#%% SOLVERS
mumps = {
    'newton_solver': {
        'linear_solver': 'mumps',
        'maximum_iterations': 50,
        'relative_tolerance': 1e-10,
        'absolute_tolerance': 1e-12,
        'relaxation_parameter': 1.0,
        'report': False,
        'error_on_nonconvergence': True,
    }
}

lu = {
    'newton_solver': {
        'linear_solver': 'lu',
        'maximum_iterations': 30,
        'relative_tolerance': 1e-8,
        'absolute_tolerance': 1e-10,
        'relaxation_parameter': 1.0,
        'report': False,
        'error_on_nonconvergence': True,
    }
}

gmres = {
    'nonlinear_solver': 'snes',
    'snes_solver': {
        'linear_solver': 'gmres',
        'preconditioner': 'ilu',
        'maximum_iterations': 50,
        'relative_tolerance': 1e-10,
        'absolute_tolerance': 1e-12,
        'line_search': 'bt',
        'report': False,
        'error_on_nonconvergence': True,
    }
}

bicgstab = {
    'nonlinear_solver': 'snes',
    'snes_solver': {
        'linear_solver': 'bicgstab',
        'preconditioner': 'ilu',
        'maximum_iterations': 50,
        'relative_tolerance': 1e-10,
        'absolute_tolerance': 1e-12,
        'line_search': 'basic',
        'report': False,
        'error_on_nonconvergence': True,
    }
}

