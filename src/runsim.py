#!/usr/bin/env python
'''
code to run a target or template simulation
'''
import sys
import os
import math
import logging
import argparse
import time

import yaml
import numpy as np
#import matplotlib.pyplot as plt 
import astropy.io.fits as fits
import bfd
import pdb

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
        description='Simulate population of target or template galaxies.\n'
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
    parser.add_argument('--outfile', '-o', help='Output moments file', type=str)
    parser.add_argument('--logfile', '-l', help='Logfile name', type=str)
    parser.add_argument('--verbose', '-v', help='Increase logging detail', action='count')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--targets', help='Make target galaxies -OR-', action='store_const', const=True)
    group.add_argument('--templates', help='Make template galaxies', action='store_const', const=True)
    
    parser.add_argument('--dir', help='Directory for files', type=str)
    parser.add_argument('--ngals', '-n', help='Number of galaxies to draw', type=int)
    parser.add_argument('--image_size', help='Pixels across postage stamps', type=int)
    parser.add_argument('--pad_factor',help='factor by which to zero pad original stamp size specified by --image_size, default=1, no padding',type=int)
    parser.add_argument('--pixel_scale', help='Sky units per pixel', type=float)
    parser.add_argument('--shear', help='Shear applied to targets', type=float, nargs=2)

    defaults = {'dir':"",
                'outfile':None,  # ??? No default for output
                'logfile':None,  # log to screen by default
                'ngals':10,
                'image_size':48,
                'pad_factor':1,
                'pixel_scale':1.,
                'shear': [0.02, 0.0] }

    group = parser.add_argument_group(title='PSF',description='PSF parameters')
    group.add_argument('--psf_type', help='PSF type', type=str)
    group.add_argument('--psf_args', help='PSF construction arguments', nargs='*', type=float)
    group.add_argument('--psf_e', help='PSF ellipticity components', nargs=2, type=float)
    defaults_psf = {'type':'Moffat',
                    'args':[1.5, 3.5],
                    'e':[0., 0.02]}

    group = parser.add_argument_group(title='Weights',description='Weight parameters')
    group.add_argument('--wt_n', help='Weight function index', type=int)
    group.add_argument('--wt_sigma', help='Weight function sigma', type=float)
    defaults_wt = {'n':4,
                   'sigma':3.5}

    group = parser.add_argument_group(title='Galaxies',description='Galaxy population parameters')
    group.add_argument('--gal_sn_range', help='Approx flux moment S/N range', nargs=2, type=float)
    group.add_argument('--gal_hlr_range', help='Half-light radius range', nargs=2, type=float)
    group.add_argument('--gal_noise_var', help='Noise variance', type=float)
    group.add_argument('--gal_e_sigma', help='Intrinsic ellipticity RMS per component', type=float)
    group.add_argument('--gal_seed', help='Galaxy generator seed', type=int)
    group.add_argument('--gal_fixsersic',help='If want to fix to be only disk (1) or bulge (2) or 2d gaussian (3), default is combination (0)', type=int)
    defaults_gal = {'sn_range':[5., 25.],
                        'hlr_range':[1.5, 3.0],
                        'noise_var':100.,
                        'e_sigma':0.2,
                        'seed':0,
                        'fixsersic':0
}


    group = parser.add_argument_group(title='Templates',description='Template replication specs')
    group.add_argument('--tmpl_target_file',
                           help='Get sigma and weight info from this target table file. '
                           'Values in file take precedence over any provided in config '
                           'or command line', type=str)
    group.add_argument('--tmpl_noise_factor', help='Noise level relative to targets', type=float)
    group.add_argument('--tmpl_sn_min', help='Lower flux moment S/N cut on targets', type=float)
    group.add_argument('--tmpl_sigma_xy', help='Measurement error on xy moments', type=float)
    group.add_argument('--tmpl_sigma_flux', help='Measurement error on flux moments', type=float)
    group.add_argument('--tmpl_sigma_max', help='Maximum sigma deviation to replicate', type=float)
    group.add_argument('--tmpl_sigma_step', help='Sigma step for template replication', type=float)
    group.add_argument('--tmpl_xy_max', help='Maximum allowed centroid for replication', type=float)
    defaults_tmpl = {'target_file':None,
                         'noise_factor':0.,   # No noise on templates by default
                         'sigma_xy': None,    # No defaults for measurement errors, must be given
                         'sigma_flux': None,  #  or obtained from a target_file
                         'sn_min': 5.,
                         'sigma_max':6.5,
                         'sigma_step':1.0,
                         'xy_max':2.}
    
    args = parser.parse_args()


    # Set up our master parameter dictionary
    params = {}
    # We require certain sub-dictionaries to be present
    params['PSF'] = {}
    params['WEIGHT'] = {}
    params['GALAXY'] = {}
    params['TEMPLATE'] = {}

    # Read YAML configuration files
    if args.config_file is not None:
        for f in args.config_file:
            params.update( yaml.load(open(f)) )
    
    # Override with any command-line options, install defaults
    install_args_and_defaults(params, args, defaults)
    install_args_and_defaults(params['PSF'], args, defaults_psf, arg_prefix='psf_')
    install_args_and_defaults(params['WEIGHT'], args, defaults_wt, arg_prefix='wt_')
    install_args_and_defaults(params['GALAXY'], args, defaults_gal, arg_prefix='gal_')
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
    return params

def check_params(params):
    ''' Check that parameters are in range, and do any other processing
    necessary, including extracting sigma parameters from a target file
    if one was given.
    '''

    # Must have an outfile specified:
    if 'outfile' not in params:
        raise Exception("Must specify an outfile")

    if np.hypot(params['shear'][0],params['shear'][1]) >= 1.:
        raise Exception('Shear cannot be >=1')
    
    if not 0 <= params['GALAXY']['fixsersic'] <= 3:
        raise Exception('Must give valid value (0-3) to fixsersic')

    # Get template parameters from a target file, if one is given
    if 'target_file' in params['TEMPLATE'] and params['make_template']:
        fitsfile = os.path.join(params['dir'],params['TEMPLATE']['target_file'])
        if os.path.isfile(fitsfile):
            hdu = fits.open(fitsfile)[0];
            M0 = bfd.Moment().M0  # Index for flux moment
            params['TEMPLATE']['sigma_xy'] = np.sqrt(hdu.header['COVMXMX'])
            params['TEMPLATE']['sigma_flux'] = np.sqrt(hdu.data[M0][M0])
            params['WEIGHT']['n'] = hdu.header['WT_N']
            params['WEIGHT']['sigma'] = hdu.header['WT_SIG']
        else:
            raise Exception("Could not access target_file " + fitsfile)


def main(params):
    # Check the parameters for sanity
    check_params(params)
    template = params['make_template']

    # define center of arrays - 2d array
    cent = int(params['image_size']/2.)
    cent = np.array([cent,cent], dtype=float)
    
    psf = bfd.define_psf(**params['PSF'])
    psfarr = bfd.return_array(psf,**params)

    # create galaxy generator - shear them only for targets
    if template:
        galaxy = bfd.GalaxyGenerator(flux_range=[1.,2.], pixel_scale = params['pixel_scale'],**params['GALAXY'])
    else:
        galaxy = bfd.GalaxyGenerator(g=params['shear'],flux_range=[1.,2.],pixel_scale = params['pixel_scale'],**params['GALAXY'])
    # create weight function
    wt = bfd.KBlackmanHarris(**params['WEIGHT'])

    # Set up noise level
    noise_var = params['GALAXY']['noise_var']

    # Set flux range on galaxy generator to produce flux moments in desired range
    # And get the covariance for targets
    # set up a galaxy with S/N = 1 (F = sqrt(sigma^2))
    gal = galaxy.nominal(flux=np.sqrt(noise_var)) 
    # create an image with 0 noise
    im = bfd.return_array(gal,
                          noise_var = 0.0,
                          use_gaussian_noise = True,
                          convolve_with_psf = True,
                          psf=psf,
                          **params)
    kdata = bfd.simpleImage(im,cent,psfarr,
                            pixel_scale=params['pixel_scale'],
                            pixel_noise=np.sqrt(noise_var),
                            pad_factor=params['pad_factor'])

    mc = bfd.MomentCalculator(kdata,wt)
    mom = mc.get_moment(0.,0.)
    cov = mc.get_covariance()
    # Flux needed to have S/N=1 on flux moment:
    # S/N_gal = b * S/N_MF
    # F/sigma = b * MF/sqrt(Cov_MF)
    # for F/Sigma = 1: b = sqrt(COV_MF)/MF
    fluxSN1 = (np.sqrt(cov[0][mom.M0,mom.M0]) / mom.even[mom.M0]) * np.sqrt(noise_var)
    #  F = fluxSN1 * SN
    galaxy.flux_range = [fluxSN1*sn for sn in params['GALAXY']['sn_range']]

    if template:
        # Reduce or eliminate noise for templates
        if 'noise_factor' in params['TEMPLATE']:
            noise_var *= params['TEMPLATE']['noise_factor']
        else:
            noise_var = 0.

    # setup classes to save results
    if template:
        tab = bfd.TemplateTable(n = params['WEIGHT']['n'],
                                sigma = params['WEIGHT']['sigma'],
                                **params['TEMPLATE'])
    else:
        # Initialize the output table
        tab = bfd.TargetTable(n = params['WEIGHT']['n'],
                              sigma = params['WEIGHT']['sigma'],
                              cov=cov)


    # loop over galaxies
    for i in xrange(params['ngals']):
        # generate galaxy, image and k-image
        gal = galaxy.sample()

        im = bfd.return_array(gal,
                              noise_var = noise_var,
                              use_gaussian_noise = True,
                              convolve_with_psf = True,
                              psf=psf,
                              **params)

        kdata = bfd.simpleImage(im,cent,psfarr,
                                pixel_scale=params['pixel_scale'],
                                pixel_noise=np.sqrt(noise_var),
                                pad_factor=params['pad_factor'])

        # start moment calculator
        m = bfd.MomentCalculator(kdata,wt,id=i,nda=1./params['ngals'])


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
            xyshift, error, msg = m.recenter()
            if error:
                logging.warning("recentering did not work for %sth galaxy: %s" %(i,msg))
                tab.addLost()
            else:
                tab.add(m.get_moment(0,0), xy=xyshift, id=i)
    # save out binary fits files
    tab.save(os.path.join(params['dir'],params['outfile']))


if __name__ == '__main__':
    params = parse_input()

    aa=time.clock()
    aaa=time.time()

    # run program to produced target/template galaxies and save their moments in a fits file
    main(params)

    bb=time.clock()
    bbb=time.time()
    logging.info("run time %s" %(bb-aa))
    logging.info("clock time %s" %(bbb-aaa))

    sys.exit(0)
