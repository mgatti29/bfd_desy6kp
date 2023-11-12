#!/usr/bin/env python

import sys
import os
import math
import logging
import argparse
import time
import pdb

import yaml
import numpy as np
import matplotlib.pyplot as plt 
import astropy.io.fits as fits
import bfd
import bfdmeds
import ngmixer

def install_args_and_defaults(params, args, defaults, arg_prefix=""):
    '''
    For each key of the defaults dictionary, put the params value in the dictionary as
    * the value in args, if there is one - apply arg_prefix to parameter name
    * else the value already present in params stays, if there is one
    * else insert the defaults value, if it's not None
    * else do nothing
    '''
    for pname in defaults.keys():
        pval = eval('args.' + arg_prefix + pname)
        if pval is None:
            if defaults[pname] is not None and pname not in params:
                params[pname] = defaults[pname]
        else:
            params[pname] = pval
    return

def parse_input():
    '''Read command-line arguments and any specified YAML config files,
    returning dictionary of parameter values.
    '''
    parser = argparse.ArgumentParser(
        description='Measure and store BFD moments of target or template galaxies in MEDS files.\n'
        'Command-line args take precedence over parameters in config files.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Config file names - do not get passed back to program.
    parser.add_argument('--config_file', '-c',
                            help='Configuration file name(s), if any, in order of increasing precedence',
                            type=str, nargs='*')
    parser.add_argument('--save_config', type=str,
                            nargs='?', const="", # Comes back with null string if given w/o argument
                            help='File in which to save configuration.  With no argument, will dump a '
                            'configuration file to stderr' ) 
    parser.add_argument('--infile', '-i', help='Input MEDS file', type=str)
    parser.add_argument('--outfile', '-o', help='Output moments file', type=str)
    parser.add_argument('--psfdir', help='Directory holding PSFEx files', type=str)
    parser.add_argument('--astrofile', help='New astrometric solution, if any', type=str)
    parser.add_argument('--logfile', '-l', help='Logfile name', type=str)
    parser.add_argument('--moffile',help='Name of MOF file if want to perform subtraction and replacement of pixels, if given will perform',type=str)
    parser.add_argument('--verbose', '-v', help='Increase logging detail', action='count')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--targets', help='Make target galaxies -OR-', action='store_const', const=True)
    group.add_argument('--templates', help='Make template galaxies', action='store_const', const=True)
    
    parser.add_argument('--dir', help='Directory for files', type=str)
    defaults = {'dir':"",
                'infile':None,  # No default for input
                'outfile':None,  # or output
                'psfdir':None,  # or psf location
                'astrofile':None,  # Astrometry built into MEDS is default
                'logfile':None, # log to screen by default
                'moffile':None} # No default MOF file

    group = parser.add_argument_group(title='Weights',description='Weight parameters')
    group.add_argument('--wt_n', help='Weight function index', type=int)
    group.add_argument('--wt_sigma', help='Weight function sigma', type=float)
    defaults_wt = {'n':4,
                   'sigma':3.5}

    group = parser.add_argument_group(title='Templates',description='Template replication specs')
    group.add_argument('--tmpl_sn_min', help='Lower flux moment S/N cut on targets', type=float)
    group.add_argument('--tmpl_sigma_xy', help='Measurement error on xy moments', type=float)
    group.add_argument('--tmpl_sigma_flux', help='Measurement error on flux moments', type=float)
    group.add_argument('--tmpl_sigma_max', help='Maximum sigma deviation to replicate', type=float)
    group.add_argument('--tmpl_sigma_step', help='Sigma step for template replication', type=float)
    group.add_argument('--tmpl_xy_max', help='Maximum allowed centroid for replication', type=float)
    defaults_tmpl = {'sigma_xy': None,    # No defaults for measurement errors, must be given
                     'sigma_flux': None,  #  or obtained from a target_file
                     'sn_min': 5.,
                     'sigma_max':6.5,
                     'sigma_step':1.0,
                     'xy_max':2.}
    
    args = parser.parse_args()


    # Set up our master parameter dictionary
    params = {}
    # We require certain sub-dictionaries to be present
    params['WEIGHT'] = {}
    params['TEMPLATE'] = {}

    # Read YAML configuration files
    if args.config_file is not None:
        for f in args.config_file:
            params.update( yaml.load(open(f)) )
    
    # Override with any command-line options, install defaults
    install_args_and_defaults(params, args, defaults)
    install_args_and_defaults(params['WEIGHT'], args, defaults_wt, arg_prefix='wt_')
    install_args_and_defaults(params['TEMPLATE'], args, defaults_tmpl, arg_prefix='tmpl_')

    if args.templates is not None:
        params['make_template'] = True
    elif args.targets is not None or 'make_template' not in params:
        # Default is to make targets
        params['make_template'] = False
    
    # After all parameters are set, save to YAML if requested
    if args.save_config is not None:
        if len(args.save_config)==0:
            # Empty arg means dump to stderr and quit
            yaml.dump(params, sys.stderr)
            sys.exit(1)
        # otherwise save to file and continue
        fout = open(args.save_config, 'w')
        yaml.dump(params, fout)
        fout.close()

    # Set up logfile if there is one
    if args.verbose is None or args.verbose==0:
        level = logging.WARNING
    elif args.verbose ==1:
        level = logging.INFO
    elif args.verbose >=2:
        level = logging.DEBUG
    if args.logfile is None:
        # Logging to screen, set level
        logging.basicConfig(level = level)
    else:
        logging.basicConfig(filename = os.path.join(params['dir'],args.logfile),
                                filemode='w',
                                level = level)
    print params ###
    return params

def check_params(params):
    ''' Check that parameters are in range, and do any other processing
    necessary, including extracting sigma parameters from a target file
    if one was given.
    '''

    # Must have an infile and outfile specified:
    if 'outfile' not in params or 'infile' not in params:
        raise Exception("Must specify an outfile and infile")


def main(params):
    # Check the parameters for sanity
    check_params(params)
    
    template = params['make_template']

    # perform MOF subtraction if MOF file given
    if 'moffile' not in params or params['moffile'] is None:
        mof_file = None
    else:
        mof_file = os.path.join(params['dir'],params['moffile'])

    if mof_file is not None:
        infilenew=params['infile'].split('.')[0] + '_mofsub.fits.fz'
        mof=ngmixer.imageio.extractor_corrector.MEDSExtractorCorrector( 
            mof_file,
            os.path.join(params['dir'],params['infile']),
            0,
            100,
            os.path.join(params['dir'],infilenew),
            reject_outliers=True,
            replace_bad=True,
            bad_flags=[
                2,     # SATURATE
                4,     # INTERP
                2048,  # SUSPECT
                8192,  # NEAREDGE
                16384, # TAPEBUMP
            ])


    if 'astrofile' not in params or params['astrofile'] is None:
        astro_file = None
    else:
        astro_file = os.path.join(params['dir'],params['astrofile'])
        
    psf_source = bfdmeds.DirectoryPsfexSource(params['psfdir'])
    
    if mof_file is not None:
        # if the mof file exists, the image file is the mof subtracted
        # also read in original to see effect of mof fit
        meds=bfdmeds.BFDMEDS(os.path.join(params['dir'],infilenew),
                             psf_source,
                             astro_file = astro_file)
        medsorig=bfdmeds.BFDMEDS(os.path.join(params['dir'],params['infile']),
                                 psf_source,
                                 astro_file = astro_file)
    else:
        # if the mof file is not input, the image file is the original
        meds=bfdmeds.BFDMEDS(os.path.join(params['dir'],params['infile']),
                             psf_source,
                             astro_file = astro_file)


    # set weight params
    wt = bfd.KBlackmanHarris(**params['WEIGHT'])

    tab = None  # Signal creation of output table
        

    # figure out number of total galaxies
    numcutouts = meds._cat['ncutout']
    numobjs = numcutouts.shape[0] # loop over each obj

    mf=np.array([])
    mmf=np.array([])
    mr=np.array([])
    mmr=np.array([])
    mp=np.array([])
    mmp=np.array([])
#    for i in xrange(numobjs):
    for i in xrange(100):

        # get image data for the object
        obji = meds.get_cutout_list(i,skip_coadd=True)
        
        if mof_file is not None:
            # if the mof file exists, read in original image
            origobji=medsorig.get_cutout_list(i,skip_coadd=True)

        nims = len(obji)

        # dictionary of WCS jacobian
        wcsi = meds.get_jacobian_list(i, skip_coadd=True)

        # background noise of image (sigma) 
        noisei = meds.get_noise_list(i, skip_coadd=True)
        pdb.set_trace()
        # list of PSF images
        psfi = meds.get_psf_list(i, skip_coadd=True)

        kdata = []
        if mof_file is not None:
            kdatar = []
        for j in range(nims):
            origin = (0.,0.)  # "World" coord galaxy center
            xyref = (wcsi[j]['col0'],wcsi[j]['row0'])  # pixel coordinate of fixed point
            uvref = (0., 0.)  # world coordinates of fixed point
            duv_dxy = np.array( [ [wcsi[j]['dudcol'], wcsi[j]['dudrow']],
                                  [wcsi[j]['dvdcol'], wcsi[j]['dvdrow']] ])
            wcs = bfd.WCS(duv_dxy, xyref=xyref, uvref=uvref)

            kdata.append( bfd.simpleImage(obji[j], origin, psfi[j], wcs=wcs, pixel_noise=noisei[j]))

            if mof_file is not None:
                # if mof file supplied, get Fourier space image of residual 
                # (original - final
                kdatar.append( bfd.simpleImage(origobji[j]-obji[j], origin, psfi[j], wcs=wcs, pixel_noise=noisei[j]))

        m = bfd.MultiMomentCalculator(kdata, wt, id = i, nda=1.0)
        if mof_file is not None:
            mr = bfd.MultiMomentCalculator(kdatar, wt, id = i, nda=1.0)

        if tab is None:
            # setup classes to save results
            if template:
                tab = bfd.TemplateTable(n = params['WEIGHT']['n'],
                                        sigma = params['WEIGHT']['sigma'],
                                        **params['TEMPLATE'])
            else:
                # Initialize the output table, don't provide covariance matrix: will save covariance matrix for each galaxy
                tab = bfd.TargetTable(n = params['WEIGHT']['n'],
                                    sigma = params['WEIGHT']['sigma'],
                                      cov=None)
                if mof_file is not None:
                    tabr = bfd.TargetTable(n = params['WEIGHT']['n'],
                                           sigma = params['WEIGHT']['sigma'],
                                           cov=None)

        # if template, save even & odd moments and derivs from iterating around
        if template:
            # run procedure to obtain templates at different coords near galaxy center
            t = m.make_templates(**params['TEMPLATE'])
            if t[0] is None:
                logging.warning(t[1] + " for %sth galaxy" %(i))
            else:                
                for tmpl in t:
                    tab.add(tmpl)
        else:
            # if target get moments at MX=MY=0 (only care about even moments)
            # save out even and odd covariances
            xyshift, error, msg = m.recenter()
            pdb.set_trace()
            if error:
                tab.addLost()
                if mof_file is not None:
                    tabr.addLost()
            else:
                tab.add(m.get_moment(0,0), xy=xyshift, id=i,covgal=m.get_covariance())
                if mof_file is not None:
                    tabr.add(mr.get_moment(xyshift[0],xyshift[1]),xy=xyshift,id=i,covgal=m.get_covariance())


    # save out binary fits files
    tab.save(os.path.join(params['dir'],params['outfile']))
    if mof_file is not None:
        tabr.save(os.path.join(params['dir'],'residtest.fits'))

if __name__ == '__main__':
    params = parse_input()
    
    aa=time.clock()

    # run program to produced target/template galaxies and save their moments in a fits file
    main(params)

    bb=time.clock()
    logging.info("run time %s" %(bb-aa))

    sys.exit(0)
