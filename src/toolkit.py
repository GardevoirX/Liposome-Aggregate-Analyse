import os


def setEnv(threadNum):

    os.environ["OMP_NUM_THREADS"] = f"{threadNum}"  # export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = f"{threadNum}"  # export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = f"{threadNum}"  # export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = (
        f"{threadNum}"  # export VECLIB_MAXIMUM_THREADS=1
    )
    os.environ["NUMEXPR_NUM_THREADS"] = f"{threadNum}"  # export NUMEXPR_NUM_THREADS=1

    return 0
