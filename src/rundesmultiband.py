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
import bfd
import bfdmeds
import multiprocessing
import copy
import astropy.io.fits as fits

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
    parser.add_argument('--infile_version',help='version of input MEDS file',type=str,default='y3v02')
    parser.add_argument('--outfile', '-o', help='Output moments file', type=str)
    parser.add_argument('--psfdir', help='Directory holding PSFEx files', type=str)
    parser.add_argument('--psftype', help='specify if psf files are PSFEx, Piff, default is None and uses psf extension of meds', type=str)
    parser.add_argument('--matchedap_file',help='BFD output file from previous run for using same RA/Dec',type=str,default=None)
    
    parser.add_argument('--astrodir', help='directory to new astrometric solution, if any', type=str)
    parser.add_argument('--logfile', '-l', help='Logfile name', type=str)
    parser.add_argument('--moffile',help='Name of MOF file if want to perform subtraction and replacement of pixels, if given will check if exists and if it does perform mof subtraction',type=str)
    parser.add_argument('--mof_subtraction',help="type of mof neighbor subtraction to perform. specify 'ngmixer' for the ngmixer extractorcorrector (using cm DES Y3 wide field cm models), or 'internal' for using BDF models",default='none',type=str)
    parser.add_argument('--bands',help='list of imaging bands for code, e.g. ["r","g","i","z"]',default=None,nargs='+')
    parser.add_argument('--band_weights',help='weight for each imaging band in same order',default=None,nargs='+')
    parser.add_argument('--make_band_weight_test_file',help='create file for testing the band weights',default='no') 
    parser.add_argument('--save_bands_separately',help='save out moments from each band separately in its own file',default='no')
    parser.add_argument('--bad_pixel_threshold',help='percentage of pixels that can be masked and replaced to be used in moment calculation. Images with perbadpix > bad_pixel_threshold will not be included. Default is 1 such that all images can be used',default=1,type=float)
    parser.add_argument('--ngals',help='number of galaxies to loop through if do not want to do all (-1 for all)',default=-1,type=int)
    parser.add_argument('--pad_factor',help='factor by which to zero pad the images',default=1,type=int)
    parser.add_argument('--num_proc',help='specifiy number of processes over which to divide objects',default=1,type=int)

    parser.add_argument('--verbose', '-v', help='Increase logging detail', action='count')
    parser.add_argument('--mode',help="mode for running code: normal or debug",default="normal")

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--targets', help='Make target galaxies -OR-', action='store_const', const=True)
    group.add_argument('--templates', help='Make template galaxies', action='store_const', const=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--skip_coadd', help='Skip the coadd (first entry in meds file) -OR-', action='store_const', const=True)
    group.add_argument('--keep_coadd', help='use the coadd', action='store_const', const=True)
    
    parser.add_argument('--dir', help='Directory for files', type=str)
    defaults = {'dir':"",
                'infile':None,  # No default for input
                'infile_version':'y3v02',
                'outfile':None,  # or output
                'psfdir':None,  # or psf location
                'psftype':None, # or psfex or piff
                'matchedap_file':None,# BFD output from previous run to apply same centers
                'astrodir':None,  # Astrometry built into MEDS is default
                'logfile':None, # log to screen by default
                'moffile':None, # No default MOF file
                'mof_subtraction':'none', # no subtraction is performed by default
                'bands':None, # list of input imaging bands
                'band_weights':None, # lists of corresponding weights
                'make_band_weight_test_file':'no', # make the test file for the band weights
                'save_bands_separately':'no', 
                'ngals':-1, # number of gals to loop through if not none
                'bad_pixel_threshold':1,
                'pad_factor':1,
                'num_proc':1,
                'mode':"normal"}

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

    if args.keep_coadd is not None or params['make_template'] is True:
        params['skip_coadd']=False
    elif args.skip_coadd is not None or 'skip_coadd' not in params:
        # Default is to skip the coadd
        params['skip_coadd']=True

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

    if params['make_band_weight_test_file'] != 'no':
        if params['make_band_weight_test_file'] != 'yes':
            raise Exception("make_band_weight_test_file must be yes or no")

    if params['save_bands_separately'] != 'no':
        if params['save_bands_separately'] != 'yes':
            raise Exception("make_band_weight_test_file must be yes or no")

    if params['mode'] != 'normal':
        if params['mode'] != 'debug':
            raise Exception("mode must be in either normal or debug")

    if params['make_template'] == True and params['skip_coadd'] == True:
        raise Exception("For templates, cannot skip_coadd")

    if 'astrodir' in params:
        os.environ["CAL_PATH"]=params["astrodir"]
        if os.path.isfile(os.path.join(os.environ['CAL_PATH'],'y4a1.guts.astro')) == False:
            raise Exception("path to astronometry files is not valid")
    
    if 'psfdir' in params:
        if 'psftype' not in params:
            raise Exception("Must give type of psf if supplying directory")

    if 'psftype' in params:
        goodpsf = (params['psftype'] == 'piff') | (params['psftype'] == 'psfex')
        if goodpsf == False:
            raise Exception("PSF type must be either piff or psfex")

    return

def loop(allinfo):

    # unpack info from allinfo
    params=allinfo['PARAMS']
    numobjtot=allinfo['FILEINFO']['numobj']
    ids=allinfo['FILEINFO']['ids']
    numbers=allinfo['FILEINFO']['numbers']
    ras=allinfo['FILEINFO']['ras']
    g_i_use=allinfo['FILEINFO']['g_i_use']
    decs=allinfo['FILEINFO']['decs']
    start=allinfo['SETUPINFO']['start']
    end=allinfo['SETUPINFO']['end']
    template=allinfo['SETUPINFO']['template']
    bands=allinfo['SETUPINFO']['bands']
    psf_source=allinfo['SETUPINFO']['psf_source']
    mof_file=allinfo['SETUPINFO']['mof_file']
    astro_dir=allinfo['SETUPINFO']['astro_dir']
    bandinfo=allinfo['SETUPINFO']['bandinfo']

    # specify nda
    if template:
        nda=1./numobjtot
    else:
        nda=1.0

    # number of objects to be done in this loop
    numobjs=end-start

    if (mof_file is not None) & (params['mof_subtraction'] == "ngmixer"):
        for j in xrange(len(bands)):
            bfd.perform_mof_correction(mof_file,bands[j],start,end,params)

    # return a list of meds files, if mof_file is given
    # meds contains the corrected images, medsorig contains original image
    # if no mof_file is given, meds contains image and medsorig is None

    meds,medsorig=bfd.return_meds_list(bands,psf_source,astro_dir,mof_file,params,start=start,end=end,colors=g_i_use)

    # set weight params
    wt = bfd.KBlackmanHarris(**params['WEIGHT'])

    # Signal creation of output table
    tab = None  
    tab_resid=None
    tab_band_weight=None
    tab_band=[]
    # if want to test the band weights run this step
    if params['make_band_weight_test_file'] == 'yes':
        tab_band_weight=bfd.make_band_weight_file(len(bands))

    if  (mof_file is not None) & (params['mof_subtraction']=="internal"):
        datamodels,cols,dataepochs,opticalmodels=bfd.read_mof_file(mof_file)
        mof_info=[datamodels,cols,dataepochs,opticalmodels]
    else:
        mof_info=None

    # check if being run in matched aperture mode (e.g. IR running after fits from riz)
    if params['matchedap_file'] is not None:
        match_data=fits.open(os.path.join(params['dir'],params['matchedap_file']))
        match_id=match_data[1].data['id']
        match_ra=match_data[1].data['xy'][:,0]
        match_dec=match_data[1].data['xy'][:,1]
        match_data=0

    for i in xrange(start,end+1):
        print i
        # determine if offset should be applied
        if params['mof_subtraction'] == "ngmixer":
            # mof correction performed first, need to offset
            offset=start
        else:
            # using original file, keep indexing
            offset=0

        if tab is None:
            # setup out put table to save results
            if template:
                tab = bfd.TemplateTable(n = params['WEIGHT']['n'],
                                        sigma = params['WEIGHT']['sigma'],
                                        **params['TEMPLATE'])
            else:
                # set up output table, don't provide covariance matrix
                # will save covariance matrix for each galaxy
                tab = bfd.TargetTable(n = params['WEIGHT']['n'],
                                      sigma = params['WEIGHT']['sigma'],
                                      cov=None)
                # setup individual tables to save results per band
                if params['save_bands_separately'] == 'yes':
                    for j in xrange(len(bands)):
                        tab_band.append(bfd.TargetTable(n=params['WEIGHT']['n'],
                                                        sigma=params['WEIGHT']['sigma'],
                                                        cov=None))
                #setup table for residual image moments if MOF file was given
                if mof_file is not None:
                    tab_resid = bfd.TargetTable(n = params['WEIGHT']['n'],
                                           sigma = params['WEIGHT']['sigma'],
                                           cov=None)
        # if using matched phot and BFD didn't work in riz, don't bother running in forced mode
        skipgal=False
        if (params['matchedap_file'] is not None): 
            if (ids[i] not in list(match_id)):
                skipgal=True
        # if using matched phot and data do not exist for particular band, then skip
            nocutouts=[mmj['ncutout'][i]==0 for mmj in meds]
            if any(nocutouts):
                skipgal=True
            if skipgal and not template:
                tab.addLost()
                continue

        # get a kdata list for all images in all bands
        kdata, kdata_resid = bfd.return_kdata_list(i,bands,meds,params,medsorig=medsorig,offset=offset,skip_coadd=params['skip_coadd'],mof_info=mof_info)

        # run through loop for template
        if template:
            # if there is no acceptible data, e.g. the images are masked beyond the set level
            # continue on
            if len(kdata) == 0:
                continue
            # setup moment calculator
            m = bfd.MultiMomentCalculator(kdata, wt, id = ids[i], nda=nda,bandinfo=bandinfo)   
            if mof_file is not None:
                m_resid = bfd.MultiMomentCalculator(kdata_resid, wt, id = ids[i], nda=nda,bandinfo=bandinfo)
            # if there does not exist at least one image from each band used then
            # save lost target and continue
            if m.error_status is not None:
                continue

            # run procedure to obtain templates at different coords near galaxy center
            # save even & odd moments and derivs from iterating around
            t = m.make_templates(**params['TEMPLATE'])
            if t[0] is None:
                logging.warning(t[1] + " for %sth galaxy" %(i))
            else:                
                for tmpl in t:
                    tab.add(tmpl)

        # run through loop for targets
        else:
            # if there is no acceptible data, e.g. the images are masked beyond the set level
            # addlost for the target and continue on
            if len(kdata) == 0:
                tab.addLost()
                if params['save_bands_separately'] == 'yes':
                    for tabb in tab_band:
                        tabb.addLost()
                continue
            # setup moment calculator
            m = bfd.MultiMomentCalculator(kdata, wt, id = ids[i], nda=nda,bandinfo=bandinfo)
            #m_bkgdsub=bfd.MultiMomentCalculator(kdata_bkgdsub,wt,id=ids[i],nda=nda,bandinfo=bandinfo)

            if mof_file is not None:
                m_resid = bfd.MultiMomentCalculator(kdata_resid, wt, id = ids[i], nda=nda,bandinfo=bandinfo)
            # if there does not exist at least one image from each band used then
            # save lost target and continue
            if m.error_status is not None:
                tab.addLost()
                if params['save_bands_separately'] == 'yes':
                    for tabb in tab_band:
                        tabb.addLost()
                if mof_file is not None:
                        tab_resid.addLost()
                continue
        
            # if want to save out band weight test file add galaxy here
            if params['make_band_weight_test_file'] == 'yes':
                tab_band_weight.add_gal(m,id=ids[i],usemoment=2)
                                        
            # get moments at MX=MY=0 (only care about even moments)
            # save out even and odd covariances
            # try to find center
            if params['matchedap_file'] is not None:
                ra0,dec0=ras[i],decs[i]
                sel=np.where(ids[i] == match_id)[0][0]
                raI,decI=match_ra[sel],match_dec[sel]
                dx_use,dy_use=((raI-ra0)*3600.,(decI-dec0)*3600.)
                xyshift=np.array([dx_use,dy_use])
                error=False
            else:
                xyshift, error, msg = m.recenter()
                dx_use,dy_use=0,0

            if error:
                tab.addLost()
                if mof_file is not None:
                    tab_resid.addLost()
                if params['save_bands_separately'] == 'yes':
                    for tabb in tab_band:
                        tabb.addLost()
            else:
                newcent=np.array([ras[i],decs[i]])+xyshift/3600.0
                #deltamf=m.get_moment(0,0).even[0]-m_bkgdsub.get_moment(xyshift[0],xyshift[1]).even[0]
                #covdeltamf=m_bkgdsub.get_covariance()[0][0,0]
                tab.add(m.get_moment(dx_use,dy_use), xy=newcent, id=ids[i],number=numbers[i],covgal=m.get_covariance(),num_exp=len(kdata))#,delta_flux=deltamf,cov_delta_flux=covdeltamf)
                if mof_file is not None:
                    tab_resid.add(m_resid.get_moment(xyshift[0],xyshift[1]),xy=newcent,id=ids[i],number=numbers[i],covgal=m.get_covariance())
                if params['save_bands_separately'] == 'yes':
                    mm,mmb=m.get_moment(dx_use,dy_use,returnbands=True)
                    cce,cco,ccbe,ccbo=m.get_covariance(returnbands=True)
                    for jj,tabb in enumerate(tab_band):
                        tabb.add(mmb[jj],xy=newcent,id=ids[i],number=numbers[i],covgal=(ccbe[:,:,jj],ccbo[:,:,jj]))

    # before returning to main program, remove temporary meds files from MOF correction (if performed)
    if (mof_file is not None) & (params['mof_subtraction']=="ngmixer"):
        for j in xrange(len(bands)):
            os.remove(os.path.join(params['dir'],'tempdir',params['infile'].split('/')[-1] + '_' + bands[j] + '_meds_mofcorr_'+str(start)+'_'+str(end)+'.fits.fz'))

    return tab,tab_resid,tab_band,tab_band_weight


def main(params):
    # Check the parameters for sanity
    check_params(params)

    # setup a temporary directory to store temporary files
    tempdir=os.path.join(params['dir'],'tempdir')
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)

    # set up band and band weight dictionary for bandinfo
    bands,bandinfo=bfd.return_bandinfo(params)

    # set up wcs and psf info
    if 'astrodir' not in params or params['astrodir'] is None:
        astro_dir = None
    else:
        astro_dir = os.environ['CAL_PATH']

    if 'psfdir' not in params or params['psfdir'] is None:
        psf_source=None
    else:
        psf_source = bfdmeds.DirectoryPsfInfoSource(params['psfdir'],params['psftype'])

    if 'matchedap_file' not in params or params['matchedap_file'] is None:
        params['matchedap_file']=None


    # figure out number of total galaxies, get id, number, ra, and dec for all gals
    # get number of objects from first meds file
    numobjs, ids, numbers, ras, decs = bfd.return_objinfo(os.path.join(params['dir'],
                                       params['infile']+'_'+bands[0]+'_meds-'+ 
                                       params['infile_version']+'.fits.fz'),
                                                psf_source)
    if params['ngals'] != -1:
        if params['ngals'] > numobjs:
            raise Exception("ngals must be <= total number of objects in file")
        numobjs=params['ngals']

    # tell whether a template or a target
    template = params['make_template']

    # define MOF file
    if 'moffile' not in params or params['moffile'] is None:
        mof_file = None
        g_i_use = 1.0
    else:
        mof_file = os.path.join(params['dir'],params['moffile'])
        g_i_use = bfd.get_g_minus_i(mof_file,ids)

    # decide how to spread over multiple processes
    ngalsperproc=numobjs//params['num_proc']
    start=[]
    end=[]
    for nc in xrange(params['num_proc']):
        start.append(nc*ngalsperproc)
        end.append((nc+1)*ngalsperproc-1)
        if nc == params['num_proc']-1:
            end[nc]=numobjs-1

    # set up dictionaries to pass info to loop function
    allinfo=[]
    allinfozero={}
    fileinfo={'ids':ids,
              'numbers':numbers,
              'ras':ras,
              'decs':decs,
              'numobj':numobjs,
              'g_i_use':g_i_use}
    setupinfo={'psf_source':psf_source,
               'mof_file':mof_file,
               'astro_dir':astro_dir,
               'template':template,
               'bands':bands,
               'start':start[0],
               'end':end[0],
               'bandinfo':bandinfo}

    allinfozero['PARAMS']=params
    allinfozero['FILEINFO']=fileinfo
    allinfozero['SETUPINFO']=setupinfo
    allinfo=[copy.deepcopy(allinfozero) for _ in xrange(params['num_proc'])]
    for nc in xrange(params['num_proc']):
        allinfo[nc]['SETUPINFO']['start']=start[nc]
        allinfo[nc]['SETUPINFO']['end']=end[nc]
    # run with pool if in normal mode, otherwise just run one loop for debug mode

    if params['mode'] == 'normal':
        pool=multiprocessing.Pool(processes=params['num_proc'])
        result=pool.map(loop,allinfo)
    else:
        result=loop(allinfo[0])
        result=[result]

    tab,tab_resid,tab_band,tab_band_weight = bfd.combine_tables(result,params['num_proc'],bands,template=template)

    # save out binary fits files
    tab.save(os.path.join(params['dir'],params['outfile']))

    if (mof_file is not None) and (template == False):
        tab_resid.save(os.path.join(params['dir'],(params['outfile'].split('.'))[0]+"_resid.fits"))

    if params['make_band_weight_test_file']=='yes':
        tab_band_weight.save_file(os.path.join(params['dir'],(params['outfile'].split('.'))[0]+"_test_band_weight.fits"))

    if params['save_bands_separately'] == 'yes':
        for jj,tabb in enumerate(tab_band):
            tabb.save(os.path.join(params['dir'],(params['outfile'].split('.'))[0]+"_"+bands[jj]+".fits"))



if __name__ == '__main__':
    params = parse_input()
    
    aa=time.clock()

    # run program to produced target/template galaxies and save their moments in a fits file
    main(params)

    bb=time.clock()
    logging.info("run time %s" %(bb-aa))

    sys.exit(0)
