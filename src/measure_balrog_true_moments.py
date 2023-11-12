#!/usr/bin/env python
'''
code to measure the truth moments of the injected Balrog galaxies
'''
import sys
import os
import math
import logging
import argparse
import time

import yaml
import numpy as np
import matplotlib.pyplot as plt 
import astropy.io.fits as fits
import bfd
import bfdmeds
import ngmix
import galsim
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
    parser.add_argument('--infile', '-i', help='Input ngmix catalog of injected Balrog galaxies', type=str)
    parser.add_argument('--outfile', '-o', help='Output moments file', type=str)
    parser.add_argument('--logfile', '-l', help='Logfile name', type=str)
    parser.add_argument('--verbose', '-v', help='Increase logging detail', action='count')


    parser.add_argument('--dir', help='Directory for files', type=str)
    parser.add_argument('--image_size', help='Pixels across postage stamps', type=int)
    parser.add_argument('--pixel_scale', help='Sky units per pixel', type=float)

    defaults = {'dir':"",
                'infile':None,
                'outfile':None,  # ??? No default for output
                'logfile':None,  # log to screen by default
                'image_size':48,
                'pixel_scale':0.2631}

    group = parser.add_argument_group(title='PSF',description='PSF parameters')
    group.add_argument('--psf_T', help='PSF size', type=str)
    group.add_argument('--psf_flux', help='PSF size', type=str)
    defaults_psf = {'T':0.29,
                    'flux':1.0}

    group = parser.add_argument_group(title='Weights',description='Weight parameters')
    group.add_argument('--wt_n', help='Weight function index', type=int)
    group.add_argument('--wt_sigma', help='Weight function sigma', type=float)
    defaults_wt = {'n':4,
                   'sigma':1.0}

    args = parser.parse_args()


    # Set up our master parameter dictionary
    params = {}
    # We require certain sub-dictionaries to be present
    params['PSF'] = {}
    params['WEIGHT'] = {}

    # Read YAML configuration files
    if args.config_file is not None:
        for f in args.config_file:
            params.update( yaml.load(open(f)) )
    
    # Override with any command-line options, install defaults
    install_args_and_defaults(params, args, defaults)
    install_args_and_defaults(params['PSF'], args, defaults_psf, arg_prefix='psf_')
    install_args_and_defaults(params['WEIGHT'], args, defaults_wt, arg_prefix='wt_')

    
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

    # Must have an infile and outfile specified:
    if 'infile' not in params:
        raise Exception("Must specify an infile")

    if 'outfile' not in params:
        raise Exception("Must specify an outfile")


def main(params):
    # Check the parameters for sanity
    check_params(params)
    
    # read in input file
    hdu=fits.open(os.path.join(params['dir'],params['infile']))
    ngals=len(hdu[1].data['fofid'])

    # create weight function
    wt = bfd.KBlackmanHarris(**params['WEIGHT'])

    # create PSF
    psf_pars = [0.0, 0.0, 0.0, 0.0, params['PSF']['T'], params['PSF']['flux']]
    gm_psf = ngmix.gmix.GMixModel(psf_pars, 'gauss')
    gs_psf = gm_psf.make_galsim_object()

    # create table to store moments
    tab = bfd.TargetTable(n = params['WEIGHT']['n'],
                          sigma = params['WEIGHT']['sigma'])


    # set up band info for BFD
    bands=['g','r','i','z']
    bandinfo={}
    bandinfo['bands']=bands
    bandinfo['weights']=np.array([0.0,0.55,0.3,0.15])
    bandinfo['index']=np.arange(len(bands))
    tab=None
    tabband=[]

    # loop over galaxies
    for i in xrange(ngals):
        # pull out info for ith galaxy
        obji=hdu[1].data[i]

        # determine size of box and create GalSim and BFD WCS
        dims = [ obji['box_size'], obji['box_size'] ]
        center=obji['box_size']/2
        cent=(center,center)
        origin = (0., 0.)
        duv_dxy = np.array( [ [0.0, params['pixel_scale']],
                              [params['pixel_scale'], 0.0] ])

        wcs_bfd = bfd.WCS(duv_dxy,xyref=cent,uvref=origin)

        wcs_gs=galsim.AffineTransform(0.0,params['pixel_scale'],params['pixel_scale'],0.0,origin=galsim.PositionD(cent[0],cent[1]),world_origin=galsim.PositionD(origin[0],origin[1]))

        # set up list to store kdata
        kdata=[]

        # draw psf with correct box size
        psfimage=galsim.ImageD(dims[0],dims[1])
        gs_psf.drawImage(psfimage,wcs=wcs_gs,method='auto') # need pixel conv
        psfarr=psfimage.array
        psfarr/=np.sum(psfarr)

        # loop through bands
        for j,band in enumerate(bands):
            objij_pars = [ 0, 0, obji['cm_g'][0], obji['cm_g'][1], obji['cm_T'], obji['cm_flux'][j] ]
            gal_gm = ngmix.gmix.GMixCM(obji['cm_fracdev'],obji['cm_TdByTe'],objij_pars)
            gal_gs = gal_gm.make_galsim_object()

            # convolution with the PSF
            cgal_gs=galsim.Convolve([gal_gs,psf_gs])
            
            # draw galaxy image
            image=galsim.ImageD(dims[0],dims[1])
            cgal_gs.drawImage(image=image,wcs=wcs_gs,method='auto') # need pixel conv
            imij = image.array


            kdata.append(bfd.simpleImage(imij,origin,psfarr,
                                         wcs=wcs_bfd,band=band))


        # start moment calculator
        m = bfd.MultiMomentCalculator(kdata,wt,id=i,nda=1./ngals,bandinfo=bandinfo)

        # start a target table if not already
        if tab is None:
            tab=bfd.TargetTable(n=params['WEIGHT']['n'],
                                sigma=params['WEIGHT']['sigma'],
                                cov=None)
            for j in xrange(len(bands)):
                tabband.append(bfd.TargetTable(n=params['WEIGHT']['n'],
                                               sigma=params['WEIGHT']['sigma'],
                                               cov=None))

        # recenter - should be drawn at center so don't think I need to do
        #xyshift, error, msg = mj.recenter()
        #if error:
        #    tab.addLost()
        #else:
        mm,mmb=m.get_moment(0,0,returnbands=True)
        tab.add(mm, id=hdu[1].data['fofid'][i])
        for j,tabb in enumerate(tabband):
            tabb.add(mmb[j],id=hdu[1].data['fofid'][i])

    # save out binary fits files
    tab.save(os.path.join(params['dir'],params['outfile']))
    for j,tabb in enumerate(tabband):
        tabb.save(os.path.join(params['dir'],(params['outfile']).split('.')[0]+'_'+bands[j]+'.fits'))

    hdu_bfd=fits.open(os.path.join(params['dir'],(params['outfile']).split('.')[0]+'_r.fits'))
    mf_r=hdu_bfd[1].data['moments'][:,0]
    flux_r=hdu[1].data['cm_flux'][:,1]
    plt.plot(flux_r,mf_r/flux_r,'b.')
    plt.show()
    pdb.set_trace()
                               

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
