import os
import sys
import inspect

curr_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
sys.path.append(curr_path)
