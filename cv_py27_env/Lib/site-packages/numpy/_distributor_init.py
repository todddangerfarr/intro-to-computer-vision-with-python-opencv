# Initialize numpy+MKL

import os

# Disable Intel Fortran default console event handler
env = 'FOR_DISABLE_CONSOLE_CTRL_HANDLER'
if env not in os.environ:
    os.environ[env] = '1'

# Prepend the path of the Intel runtime DLLs to os.environ['PATH']
try:
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'core')
    if path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = os.pathsep.join((path,
                                              os.environ.get('PATH', '')))
except Exception:
    pass

NUMPY_MKL = True
