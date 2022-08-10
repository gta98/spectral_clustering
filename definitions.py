
from enum import Enum

CHANGE_TO_TRUE_BEFORE_ASSIGNMENT = False

FLAG_PRODUCTION = CHANGE_TO_TRUE_BEFORE_ASSIGNMENT
FLAG_DEBUG = not FLAG_PRODUCTION
FLAG_VERBOSE_PRINTS = True and FLAG_DEBUG
FLAG_VERBOSE_ERRORS = True and FLAG_DEBUG

MSG_ERR_INVALID_INPUT = "Invalid Input!"
MSG_ERR_GENERIC = "An Error Has Occurred"

INFINITY = float('inf')
JACOBI_MAX_ROTATIONS = 100
JACOBI_EPSILON = 1e-5
KMEANS_EPSILON = 1e-5 # FIXME - what should this be? 0?
KMEANS_MAX_ITER = 300 # this was verified to be 300

class InvalidInputTrigger(ValueError): pass
class GenericErrorTrigger(Exception): pass

class Goal(Enum):
    WAM = wam = 1
    DDG = ddg = 2
    LNORM = lnorm = 3
    JACOBI = jacobi = 4
    SPK = spk = 5