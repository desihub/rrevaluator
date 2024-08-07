"""
rrevaluator.util
================

General utilities.

"""
import numpy as np

from desiutil.log import get_logger
log = get_logger()

C_LIGHT = 299792.458 # [km/s]

def getstats(z, ztrue, zwarn, vcut=3e3, indices=False, good=False):
    """
    * Purity = what fraction of redrock confident answers were actually correct?  i.e. #(goodz AND ZWARN=0)/#(ZWARN=0)
    * Completeness = what fraction of all good VI entries had correct and confident redshifts? i.e. #(goodz AND ZWARN=0)/#(total)
    * Catastrophic outliers = minimize number of wrong answers incorrectly flagged as good, i.e. #(badz AND ZWARN=0) / #(ZWARN=0)

    """
    Ntotal = len(z)
    if Ntotal == 0:
        return 0, np.nan, np.nan, np.nan

    dz = z - ztrue
    goodz = (C_LIGHT * np.abs(dz) / (1. + ztrue)) < vcut
    if good:
        return np.where(goodz)[0]
        
    badz = np.logical_not(goodz)
    nozwarn = zwarn == 0

    Nnozwarn = np.sum(nozwarn)

    if indices:
        Ipurity = np.where(goodz * nozwarn)[0]
        Icomplete = np.where(goodz * nozwarn)[0]
        Ioutliers = np.where(badz * nozwarn)[0]
        return Ntotal, Ipurity, Icomplete, Ioutliers        
    else:
        fpurity = np.sum(goodz * nozwarn) / Nnozwarn
        fcomplete = np.sum(goodz * nozwarn) / Ntotal
        foutliers = np.sum(badz * nozwarn) / Nnozwarn
        return Ntotal, fpurity, fcomplete, foutliers


def zstats(z, ztrue, vcut=1e3):
    N = len(z)
    if N == 0:
        return N, np.array([])
    dz = z - ztrue
    Idoom = np.where((C_LIGHT * np.abs(dz) / (1. + ztrue)) > vcut)[0]
            
    return N, Idoom
