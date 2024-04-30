"""
rrevaluator.io
==============

I/O tools.

"""
import os, pdb
import fitsio
import numpy as np
from glob import glob
from astropy.table import Table, vstack, join

from desiutil.log import get_logger
log = get_logger()

#projectdir = os.path.join(os.getenv('DESI_ROOT'), 'users', 'ioannis', 'Y3-templates')
projectdir = os.path.expandvars(os.path.join(os.getenv('PSCRATCH'), 'Y3-templates'))


def _tractorphot_onebrick(args):
    """Multiprocessing wrapper."""
    return tractorphot_onebrick(*args)


def tractorphot_onebrick(cat, RACOLUMN='TARGET_RA', DECCOLUMN='TARGET_DEC'):
    """Simple wrapper on desispec.io.photo.gather_tractorphot."""
    from desispec.io.photo import gather_tractorphot
    tractorphot = gather_tractorphot(cat, racolumn=RACOLUMN, deccolumn=DECCOLUMN)
    return tractorphot


def read_vi(quality=2.5, mp=1, vi_spectype=None, samplefile=None, veto_fibers=True, main=False):
    """See https://data.desi.lbl.gov/doc/releases/edr/vac/vi/

    """
    if samplefile is not None and os.path.isfile(samplefile):
        sample = Table(fitsio.read(samplefile))
        log.info(f'Read {len(sample):,d} objects from {samplefile}')
        return sample
    
    allvi = []
    for targ in ['BGS', 'LRG', 'ELG', 'QSO']:
        vifile = f'/global/cfs/cdirs/desi/public/edr/vac/edr/vi/v1.0/EDR_VI_{targ}_v1.csv'
        vi = Table.read(vifile)
        vi['TARGETCLASS'] = targ # os.path.basename(vifile).replace('EDR_VI_', '').replace('_v1.csv', '')
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
    if veto_fibers:
        veto = np.loadtxt('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/unique_badfibers.txt')
        I = np.isin(allvi['FIBER'], veto)
        log.info(f'Vetoing {np.sum(I):,d}/{len(allvi):,d} targets on bad fibers.')
        allvi = allvi[~I]

    # trim to main targets only
    if main:
        if False:
            from desitarget.io import read_targets_in_tiles
    
            log.info('Building main target list (this will take a little while).')
    
            tiles_edr = Table(fitsio.read('/global/cfs/cdirs/desi/public/edr/spectro/redux/fuji/tiles-fuji.fits'))
            tiles_sv = tiles_edr[((tiles_edr['SURVEY']=='sv3') | (tiles_edr['SURVEY']=='sv1')) & (tiles_edr['PROGRAM']!='backup')]
            tiles_sv['RA'], tiles_sv['DEC'] = tiles_sv['TILERA'], tiles_sv['TILEDEC']
    
            hpdirname = '/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/targets/main/resolve/dark'
            main_dark = Table(read_targets_in_tiles(hpdirname, tiles=tiles_sv, quick=True))
            assert(np.unique(main_dark["TARGETID"]).size == len(main_dark))
    
            hpdirname = '/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/targets/main/resolve/bright'
            main_bright = Table(read_targets_in_tiles(hpdirname, tiles=tiles_sv, quick=True))
            assert(np.unique(main_bright["TARGETID"]).size == len(main_bright))
    
            in_dark = np.isin(allvi['TARGETID'], main_dark['TARGETID'])
            in_bright = np.isin(allvi['TARGETID'], main_bright['TARGETID'])
            imain = in_dark | in_bright

            log.info(f'Trimming to {np.sum(I):,d}/{len(allvi):,d} main-survey targets.')
            allvi = allvi[imain]            
        else:
            import astropy.units as u
            from astropy.coordinates import SkyCoord
            from desispec.io.photo import gather_tractorphot
            from desitarget.cuts import select_targets
            from desitarget.targetmask import desi_mask
            from desiutil.brick import brickname as get_brickname

            if mp > 1 and 'NERSC_HOST' in os.environ:
                import multiprocessing
                multiprocessing.set_start_method('spawn')

            if samplefile is None:
                import tempfile
                _, tractorfile = tempfile.mkstemp(suffix='.fits', dir='/tmp')
            else:
                from desispec.io.util import replace_prefix
                tractorfile = replace_prefix(samplefile, 'sample', 'tractor')

            if os.path.isfile(tractorfile):
                tractor = Table(fitsio.read(tractorfile))
                log.info(f'Read Tractor photometry for {len(tractor):,d} objects from {tractorfile}')
            else:
                log.info(f'Gathering Tractor photometry for {len(allvi):,d} objects.')
                #tractor = gather_tractorphot(allvi)
    
                bricknames = get_brickname(allvi['TARGET_RA'], allvi['TARGET_DEC'])
                mpargs = []
                for brick in set(bricknames):
                    I = np.where(brick == bricknames)[0]
                    mpargs.append([allvi[I]])
    
                if mp > 1:
                    with multiprocessing.Pool(mp) as P:
                        tractor = P.map(_tractorphot_onebrick, mpargs)
                else:
                    tractor = [_tractorphot_onebrick(mparg) for mparg in mpargs]
                tractor = vstack(tractor)
    
                tractor.write(tractorfile, overwrite=True)
                log.info(f'Wrote {tractorfile}')

            log.info(f'Running target selection on {len(tractor):,d} objects.')
            alltargets = Table(select_targets(tractorfile, backup=False, numproc=1))

            ilrg = alltargets['DESI_TARGET'] & desi_mask.LRG != 0
            ielg = alltargets['DESI_TARGET'] & desi_mask.ELG != 0
            iqso = alltargets['DESI_TARGET'] & desi_mask.QSO != 0
            ibgs = alltargets['DESI_TARGET'] & desi_mask.BGS_ANY != 0
            imain = np.where(np.logical_or.reduce((ilrg, ielg, iqso, ibgs)))[0]
            targets = alltargets[imain]

            ## match on coordinates
            #rad = 0.5 * u.arcsec
            #c_allvi = SkyCoord(ra=allvi['TARGET_RA']*u.deg, dec=allvi['TARGET_DEC']*u.deg)
            #c_targets = SkyCoord(ra=targets['RA']*u.deg, dec=targets['DEC']*u.deg)
            #indx_allvi, indx_targets, d2d, _ = c_targets.search_around_sky(c_allvi, rad)
            nall = len(allvi)
            allvi = join(allvi, targets)

            log.info(f'Trimming to {len(allvi):,d}/{nall:,d} main-survey targets.')

            #allvi = allvi[indx_allvi]
            #targets = targets[indx_targets]
            #targets.rename_column('TARGETID', 'TARGETID_MAIN')
            #allvi = hstack((allvi, targets))

    # write out
    if samplefile is not None:
        log.info(f'Writing {len(allvi):,d} objects to {samplefile}')        
        allvi.write(samplefile, overwrite=True)
        
    return allvi


def read_iron_main_subset(samplefile=None, veto_fibers=True):
    """Read the subset of Iron/main tiles identified by Jaime.

    https://desisurvey.slack.com/archives/C05QG3VBBPB/p1709747811273839

    I've created a sample of TILEIDs starting with tiles-iron.fits and then
    applying some cuts. These cuts divide the tiles into three ranges (low,
    middle, high) based on percentiles for three parameters: 'EXPTIME',
    'EFFTIME_SPEC', and 'TILEDEC'. These ranges are applied to both bright and
    dark programs, resulting in 27 permutations for each program and a total of
    54 tiles.  The file I'm sharing contains two columns: TILEID and
    NTILE_SIMILAR. Here, NTILE_SIMILAR indicates the count of tiles falling
    within a similar range determined by the specified criteria. For example,
    this could represent the number of tiles with high 'EXPTIME', low
    'EFFTIME_SPEC', and high 'TILEDEC'.

    """
    def _build_sample():
        specprod = 'iron'
    
        tilefile = os.path.join(projectdir, 'sample', 'tiles-iron-main-subset.csv')
        tilelist = Table.read(tilefile)
        log.info(f'Read {len(tilelist):,d} tiles from {tilefile}')
    
        reduxdir = os.path.join(os.getenv('DESI_ROOT'), 'spectro', 'redux', specprod, 'tiles', 'cumulative')
    
        sample = []
        for tileid in tilelist['TILEID']:
            for petal in range(10):
                coaddfile = glob(os.path.join(reduxdir, str(tileid), '*', f'coadd-{petal}-{tileid}-thru*.fits'))
                if len(coaddfile) > 0:
                    coaddfile = coaddfile[0]
                    objtypes = fitsio.read(coaddfile, ext='FIBERMAP', columns='OBJTYPE')
                    rows = np.where(objtypes == 'TGT')[0]
                    fm = Table(fitsio.read(coaddfile, ext='FIBERMAP', columns=['TARGETID', 'TILEID', 'PETAL_LOC', 'FIBER', 'TARGET_RA', 'TARGET_DEC'], rows=rows))
                    log.info(f'Read {len(fm):,d} objects from {coaddfile}')            
                    sample.append(fm)
        sample = vstack(sample)
        return sample
    
    if samplefile is not None and os.path.isfile(samplefile):
        sample = Table(fitsio.read(samplefile))
        log.info(f'Read {len(sample):,d} objects from {samplefile}')
    else:
        sample = _build_sample()
        if samplefile is not None:
            log.info(f'Writing {len(sample):,d} objects to {samplefile}')        
            sample.write(samplefile, overwrite=True)

    # veto bad fibers
    if veto_fibers:
        veto = np.loadtxt('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/unique_badfibers.txt')
        I = np.isin(sample['FIBER'], veto)
        log.info(f'Vetoing {np.sum(I):,d}/{len(sample):,d} targets on bad fibers.')
        sample = sample[~I]

    return sample
