'''unit tests for python bfd modules'''

from   __future__      import division, print_function 
import os 
from pytest import approx
import pytest 
import astropy.io.fits as fits
import numpy as np
import bfd
import pdb

'''functions used later'''
def setupGalaxy(gen_input, gal_input, psf_input, wt_input):

    galaxy=bfd.GalaxyGenerator(pixel_scale=gen_input['pixel_scale'],**gal_input)
    psf=bfd.define_psf(**psf_input)

    wt=bfd.KBlackmanHarris(**wt_input)

    return galaxy.sample(), psf, wt

def drawGalaxy(gen_input, gal_input, gal, psf):

    return 

def computeCovarianceFromMeasurements(mom,N):
    Nmeas=np.size(mom)/N
    cov=np.zeros((N,N))
    for i in xrange(N):
        meani=np.mean(mom[i,:])
        for j in xrange(N):
            meanj=np.mean(mom[j,:])
            
            cov[i,j]=(1./(Nmeas-1)) * np.sum((mom[i,:]-meani)*(mom[j,:]-meanj))

    return cov


def testCovariance(gen_input, gal_input, gal, psf, wt, ndraw, recenterfirst=False):
    cent=[gen_input['image_size']/2.,gen_input['image_size']/2.]
    momevenN=np.zeros((5,ndraw))
    momoddN=np.zeros((2,ndraw))
    # first draw psf
    psfarr=bfd.return_array(psf,gen_input['pixel_scale'],gen_input['image_size'])
    for i in xrange(ndraw):
        im=bfd.return_array(gal,gen_input['pixel_scale'], gen_input['image_size'], noise_var=gal_input['noise_var'],use_gaussian_noise=True, convolve_with_psf=True,psf=psf)

        kdata=bfd.simpleImage(im,cent,psfarr,pixel_scale=gen_input['pixel_scale'],pixel_noise=np.sqrt(gal_input['noise_var']))

        mc=bfd.MomentCalculator(kdata,wt)

        if i == 0:
            if recenterfirst == True:
                xyshift, error, hm = mc.recenter()
            momevenN[:,i] = mc.get_moment(0,0).even
            momoddN[:,i]  = mc.get_moment(0,0).odd
            cov=mc.get_covariance()
        else:
            if recenterfirst == True:
                momevenN[:,i] = mc.get_moment(xyshift[0],xyshift[1]).even
                momoddN[:,i]  = mc.get_moment(xyshift[0],xyshift[1]).odd
            else:
                momevenN[:,i] = mc.get_moment(0,0).even
                momoddN[:,i]  = mc.get_moment(0,0).odd

    coveven = computeCovarianceFromMeasurements(momevenN[:,1:],5)
    covodd = computeCovarianceFromMeasurements(momoddN[:,1:],2)
    print("COV:   ratio expected       test")
    print("MFxMF: %s %s %s" %(cov[0][0][0]/coveven[0,0],cov[0][0][0],coveven[0,0]))
    print("MFxMR: %s %s %s" %(cov[0][0][1]/coveven[0,1],cov[0][0][1],coveven[0,1]))
    print("MFxM1: %s %s %s" %(cov[0][0][2]/coveven[0,2],cov[0][0][2],coveven[0,2]))
    print("MFxM2: %s %s %s" %(cov[0][0][3]/coveven[0,3],cov[0][0][3],coveven[0,3]))
    print("MFxMC: %s %s %s" %(cov[0][0][4]/coveven[0,4],cov[0][0][4],coveven[0,4]))
    print(" ")                                                                 
    print("MRxMR: %s %s %s" %(cov[0][1][1]/coveven[1,1],cov[0][1][1],coveven[1,1]))
    print("MRxM1: %s %s %s" %(cov[0][1][2]/coveven[1,2],cov[0][1][2],coveven[1,2]))
    print("MRxM2: %s %s %s" %(cov[0][1][3]/coveven[1,3],cov[0][1][3],coveven[1,3]))
    print("MRxMC: %s %s %s" %(cov[0][1][4]/coveven[1,4],cov[0][1][4],coveven[1,4]))
    print(" ")                                                                 
    print("M1xM1: %s %s %s" %(cov[0][2][2]/coveven[2,2],cov[0][2][2],coveven[2,2]))
    print("M1xM2: %s %s %s" %(cov[0][2][3]/coveven[2,3],cov[0][2][3],coveven[2,3]))
    print("M1xMC: %s %s %s" %(cov[0][2][4]/coveven[2,4],cov[0][2][4],coveven[2,4]))
    print(" ")                                                                 
    print("M2xM2: %s %s %s" %(cov[0][3][3]/coveven[3,3],cov[0][3][3],coveven[3,3]))
    print("M2xMC: %s %s %s" %(cov[0][3][4]/coveven[3,4],cov[0][3][4],coveven[3,4]))
    print(" ")                                                                 
    print("MCxMC: %s %s %s" %(cov[0][4][4]/coveven[4,4],cov[0][4][4],coveven[4,4]))
    print(" ")
    print("MXxMX: %s %s %s" %(cov[1][0][0]/covodd[0,0],cov[1][0][0],covodd[0,0]))
    print("MXxMY: %s %s %s" %(cov[1][0][1]/covodd[0,1],cov[1][0][1],covodd[0,1]))
    print(" ")
    print("MYxMY: %s %s %s" %(cov[1][1][1]/covodd[1,1],cov[1][1][1],covodd[1,1]))

    pdb.set_trace()


if __name__ == '__main__':

    gen_input={'pixel_scale':0.2,
               'image_size':48}

    gal_input={'e_sigma':0.2,
           'flux_range':[1000,1001],
           'hlr_range':[2.0,2.1],
           'noise_var': 10,
           'fixsersic': 0}

    psf_input={'type':'Moffat',
              'args':[1.5,3.5],
              'e':[0.0,0.0]}
    
    wt_input={'wt_n':4,
              'wt_sigma':3.5}

    ndraw=1001
    gal, psf, wt=setupGalaxy(gen_input, gal_input, psf_input, wt_input)
    testCovariance(gen_input, gal_input, gal, psf, wt, ndraw,recenterfirst=False)
