"""
rrevaluator.util
================

General utilities.

"""
import numpy as np

from desilutil.log import get_logger
log = get_logger()

def zstats(z, ztrue, vcut=1e3):
    N = len(z)
    if N == 0:
        return N, np.array([])
    dz = z - ztrue
    Idoom = np.where((C_LIGHT * np.abs(dz) / (1. + ztrue)) > vcut)[0]
            
    return N, Idoom
