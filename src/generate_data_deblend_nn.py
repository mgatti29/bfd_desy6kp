#!/usr/bin/env python
'''
code to run a target or template simulation
'''
import sys
import os
import logging
import argparse
import time

import yaml
import numpy as np
import matplotlib.pyplot as plt
import bfd
import galsim
import pdb
import pickle

def parse_input():
    '''Read command-line arguments and any specified YAML config files,
    returning dictionary of parameter values.
    '''
    parser = argparse.ArgumentParser(
        description='Simulate population of target or template galaxies.\n'
        'Command-line args take precedence over parameters in config files.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--n_stamps', '-n', help='Number of galaxies to draw', type=int,default=10)
    parser.add_argument('--image_size', help='Pixels across postage stamps', type=int,default=48)
    parser.add_argument('--pixel_scale', help='Sky units per pixel', type=float,default=1.0)

    group = parser.add_argument_group(title='PSF',description='PSF parameters')
    group.add_argument('--psf_type', help='PSF type', type=str,default='Moffat')
    group.add_argument('--psf_args', help='PSF construction arguments', nargs='*', type=float,default=[1.5,3.5])
    group.add_argument('--psf_e', help='PSF ellipticity components', nargs=2, type=float,default=[0.0,0.0])

    group = parser.add_argument_group(title='Weights',description='Weight parameters')
    group.add_argument('--wt_n', help='Weight function index', type=int,default=4)
    group.add_argument('--wt_sigma', help='Weight function sigma', type=float,default=3.5)

    group = parser.add_argument_group(title='Galaxies',description='Galaxy population parameters')
    group.add_argument('--gal_sn_range', help='Approx flux moment S/N range', nargs=2, type=float,default=[5.,25.])
    group.add_argument('--gal_hlr_range', help='Half-light radius range', nargs=2, type=float,default=[1.5,3.0])
    group.add_argument('--gal_noise_var', help='Noise variance', type=float,default=100.)
    group.add_argument('--gal_e_sigma', help='Intrinsic ellipticity RMS per component', type=float,default=0.2)
    group.add_argument('--gal_seed', help='Galaxy generator seed', type=int,default=12345)
    group.add_argument('--gal_fixsersic',help='If want to fix to be only disk (1) or bulge(2), default is combination (0)', type=int,default=0)
    
    args = parser.parse_args()

    return args

def make_images(central, neighbor, psf, image_size, pixel_scale, neighbor_offset):

    central_conv = galsim.Convolve([central,psf])
    neighbor_conv = galsim.Convolve([neighbor,psf])

    centralim = central_conv.drawImage(nx=image_size, ny=image_size,scale=pixel_scale,use_true_center = False, method = 'auto')
    psfim = psf.drawImage(nx=image_size, ny=image_size, scale=pixel_scale, use_true_center=False, method='auto')
    neighborim=neighbor_conv.drawImage(nx=image_size, ny=image_size,scale=pixel_scale,use_true_center = False, method = 'auto',offset=neighbor_offset)

    return centralim.array, neighborim.array, psfim.array

def get_moments(galimage, psfimage, wt, pixel_scale, fit_center=True, center=(0.,0.)):
    origin = (np.shape(galimage)[0]/2.,np.shape(galimage)[1]/2.)
    kdata = bfd.simpleImage(galimage,origin,psfimage,pixel_scale=pixel_scale)
    m = bfd.MomentCalculator(kdata, wt)
    if fit_center:
        xyshift,error,msg = m.recenter()
        moments_even=m.get_moment(0,0).even
        moments_odd=m.get_moment(0,0).odd
    else:
        moments_even=m.get_moment(center[0],center[1]).even
        moments_odd=m.get_moment(center[0],center[1]).odd

    moments=np.zeros(7)
    moments[0:5]=moments_even
    moments[5:7]=moments_odd
    if fit_center:
        return moments, xyshift
    else:
        return moments

def get_gal(galaxy):
    return galaxy.sample()

def get_neighbor_offset(image_size, pixel_scale, distmax=None):
    # calc on unit circle

    if distmax is not None:
        maxrad = distmax
    else:
        maxrad = image_size/2.
    
    theta      = np.random.rand()*(2*np.pi)
    rr         = np.random.rand()*maxrad
    dx        = rr * np.cos(theta)
    dy        = rr * np.sin(theta) 
    offset = [dx,dy]
    return offset

def main(params):

    central_moments=np.zeros((params.n_stamps,7))
    neighbor_moments=np.zeros((params.n_stamps,7))
    central_plus_neighbor_moments=np.zeros((params.n_stamps,7))
    images = np.zeros((params.n_stamps,params.image_size,params.image_size))

    # setup GALAXY Dictionary
    flux_range=np.array(params.gal_sn_range)*np.sqrt(params.gal_noise_var)
    GALAXY={'flux_range' :flux_range,
            'hlr_range'  :params.gal_hlr_range,
            'noise_var'  :params.gal_noise_var,
            'e_sigma'    :params.gal_e_sigma,
            'pixel_scale':params.pixel_scale,
            'seed'       :params.gal_seed,
            'fixsersic' :params.gal_fixsersic}
    # setup galaxy generator
    galaxy = bfd.GalaxyGenerator(**GALAXY)

    # setup PSF
    PSF = {'args':params.psf_args,
           'type':params.psf_type,
           'e':params.psf_e}
    psf = bfd.define_psf(**PSF)

    # setup WT
    WEIGHT = {'n':params.wt_n,
              'sigma':params.wt_sigma}
    wt = bfd.KBlackmanHarris(**WEIGHT)

    for i in xrange(params.n_stamps):
        central = get_gal(galaxy)
        neighbor = get_gal(galaxy)

        neighbor_offset = get_neighbor_offset(params.image_size,params.pixel_scale)

        central_im, neighbor_im, psf_im = make_images(central, neighbor, psf, params.image_size, params.pixel_scale, neighbor_offset)

        central_moments[i,:], xyoff = get_moments(central_im, psf_im, wt, params.pixel_scale, fit_center=True)
        central_momentstest = get_moments(central_im, psf_im, wt, params.pixel_scale, fit_center=False,center=xyoff)
        neighbor_moments[i,:] = get_moments(neighbor_im, psf_im, wt, params.pixel_scale, fit_center=False, center=xyoff)
        central_plus_neighbor_moments[i,:] = get_moments(central_im+neighbor_im, psf_im, wt, params.pixel_scale, fit_center=False, center=xyoff)
        
        #plt.imshow(central_im+neighbor_im)
        #plt.show()
        images[i,:,:]=central_im+neighbor_im

    data = {'images':images,
            'central_moments':central_moments,
            'neighbor_moments':neighbor_moments,
            'central_plus_neighbor_moments':central_plus_neighbor_moments}

    return data
        
# use galsimgen to create galaxies
# decide offset for neighbor galaxy
# no noise (can set in neural network)
# draw central only -> measure moments
# draw neighbor only -> measure moments (at same location)
# draw central+neighbor -> measure moments (at same location)
# check that moments subtract out
# save out blended image, all moment combos

#give neural network confused image
#ask it to give back moment of central galaxy
#ask it to give back moment of neighbor galaxy

if __name__ == '__main__':
    params = parse_input()

    aa=time.clock()
    aaa=time.time()

    # run program to produced target/template galaxies and save their moments in a fits file
    data = main(params)

    pickle.dump(data,open( "save_data.p", "wb" ))
    bb=time.clock()
    bbb=time.time()
    print("run time %s" %(bb-aa))
    print("clock time %s" %(bbb-aaa))

    sys.exit(0)
