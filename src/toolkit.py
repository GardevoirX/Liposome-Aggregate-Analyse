import pickle
import os

def setEnv(threadNum):

    os.environ["OMP_NUM_THREADS"] = "{}".format(threadNum) # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "{}".format(threadNum) # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "{}".format(threadNum) # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "{}".format(threadNum) # export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"] = "{}".format(threadNum) # export NUMEXPR_NUM_THREADS=1
    
    return 0
    
