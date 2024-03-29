"""
rrevaluator.io
==============

I/O tools.

"""
import numpy as np
from astropy.table import Table

from desiutil.log import get_logger
log = get_logger()

def read_vi(vi_spectype=None, quality=2.5):
    # see https://data.desi.lbl.gov/doc/releases/edr/vac/vi/
    from glob import glob

    #vifiles = glob('/global/cfs/cdirs/desi/public/edr/vac/edr/vi/v1.0/*.csv')
    allvi = []
    for targ in ['BGS', 'LRG', 'ELG', 'QSO']:
        vifile = f'/global/cfs/cdirs/desi/public/edr/vac/edr/vi/v1.0/EDR_VI_{targ}_v1.csv'
        vi = Table.read(vifile)
        vi['SUFFIX'] = targ # os.path.basename(vifile).replace('EDR_VI_', '').replace('_v1.csv', '')
        allvi.append(vi)
    allvi = vstack(allvi)
    I = np.where(allvi['VI_QUALITY'] >= quality)[0]
    log.info(f'Trimming to {len(I):,d}/{len(allvi):,d} VI redshifts.')
    allvi = allvi[I]

    _, uindx = np.unique(allvi['TARGETID'], return_index=True)
    log.info(f'Trimming to {len(uindx):,d}/{len(allvi):,d} unique targets.')
    allvi = allvi[uindx]

    if vi_spectype:
        I = allvi['VI_SPECTYPE'] == vi_spectype
        log.info(f'Trimming to {np.sum(I):,d}/{len(allvi):,d} objects with VI_SPECTYPE={vi_spectype}.')
        allvi = allvi[I]

    # veto bad fibers
    veto = np.loadtxt('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/unique_badfibers.txt')
    I = np.isin(allvi['FIBER'], veto)
    log.info(f'Vetoing {np.sum(I):,d}/{len(allvi):,d} bad fibers.')
    allvi = allvi[~I]

    return allvi


