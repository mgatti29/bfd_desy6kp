'''series of tests for new momentcalc.py
uses same galaxy generator parameters as wrapper.py'''

import sys
import os
import math
import logging
import pdb

import numpy as np
import matplotlib.pyplot as plt 
import astropy.io.fits as fits
import galsimgen
import momentcalc
import time


# set up all params
## recenter the galaxies for moment calculation?
recenter = False 
## choose tests
testmoments = False           # test moments with previous calc
testcovariancematrix = False  # test covariance matrix with previous calc
testrotation = False           # test rotation conforms with expected
test1stderivwrtshear = False   # test 1st deriv wrt shear (comp and finite test poss)
test2ndderivwrtshear = False   # test 2nd deriv wrt shear (comp and finite test poss)
test1stderivwrtmag = True     # test 1st deriv wrt mag (finite test poss)
test2ndderivwrtmag = True     # test 2nd deriv wrt mag (finite test poss)
test2ndderivwrtshearandmag = True # test 2nd deriv wrt shear & mag (finite test poss)
##decide which tests 
compwitholdvals = False # comparison to old values (recenter = True required)
finitedifftest=True     # finite difference test for derivatives (recenter = False preferred)

## General
template = True # True for templates, False for targets
pixel_scale = 1.0 # 1.0arcsec/pixel
stamp_size = 48
ngal = 1

## PSF
psf_type = "Gaussian" # also "Airy", "Moffat"
psf_size = 3.0 # Gaussian: FWHM in pixels, Moffat: hlr in pixels
psf_beta = 3.5 # for Moffat profile
psf_ell  = [0.0,0.0] # include ellipticity in psf if desired

## Weight Function
weight_n     = 4.0
weight_sigma = 1.5

## Galaxy
gal_ellip_sigma = 0.2
flux_range=[1000.0,5000.0]
hlr_range =[1.5,3.0]
noise_var = 0.0#100.0 
g1 = 0.0
g2 = 0.0
eseed = 777 # set a seed for ellipticity generator (0 if want to change)
gseed = 100 # set a seed for galaxy generator (0 if want to change)

## Make an empty Moment with a short name to use for writing indices
m = momentcalc.Moment()

## make psf
psf = galsimgen.define_psf(psf_type,psf_size,psf_beta,addell=False,e=psf_ell) 
psf_arr = galsimgen.return_array(psf,pixel_scale,stamp_size)

## create galaxy generator and generate one galaxy and the PSF convolved image
galaxy = galsimgen.GalaxyGenerator(gal_ellip_sigma, flux_range,hlr_range,noise_var,g1,g2,eseed,gseed)

gal = galaxy.sample()

image_arr = galsimgen.return_array(gal,pixel_scale,stamp_size,noise_level = galaxy.noise_var, use_gaussian_noise = True,convolve_with_psf = True, psf = psf)

image_arr_wnoise = galsimgen.return_array(gal,pixel_scale,stamp_size,noise_level = 100.0, use_gaussian_noise = True, noise_seed=555, convolve_with_psf = True, psf = psf)

######TEST simpleImage#######
'''
test is for simpleImage which generates the k-space 2D image
and k information for calculating the moments
want to make sure that get the same result with rfft2 as we obtained with fft2
'''

## use simpleImage to return PSF deconvolved k-space image
kimage_arr, kx, ky, d2k, conjugate, kvar = momentcalc.simpleImage(image_arr,24,24,psf_arr,pixel_scale=pixel_scale,pixel_noise=np.sqrt(noise_var))

kimage_arr_wnoise, kx_wnoise, ky_wnoise, d2k_wnoise, conjugate_wnoise, kvar_wnoise = momentcalc.simpleImage(image_arr_wnoise,24,24,psf_arr,pixel_scale=pixel_scale,pixel_noise=np.sqrt(100.0))

## use conjugate to get correct weighting of k-space array which is half the full array
wt = np.where(conjugate,2.,1.)

## use fft2 as before to create same thing
check = np.fft.fftshift(np.fft.fft2(image_arr))
psfcheck=np.fft.fftshift(np.fft.fft2(psf_arr))
psfcheck /= psfcheck[24,24]
kimage_arr_old = check/psfcheck

## check that rough flux approx is same
#print("Test Total Flux using rfft2 vs. fft2)")
#print np.sum(kimage_arr*wt).real, np.sum(kimage_arr_old).real
## also look at imaginary parts
#print np.sum(kimage_arr*wt).imag, np.sum(kimage_arr_old).imag
## not sure why not the same?


######Play with class KSigmaWeight#######

weight = momentcalc.KBlackmanHarris(weight_n,weight_sigma)
weight.set_k(kx,ky)

plt.plot(kx[0,:],weight.w[0,:],'b')
plt.plot(kx[0,:],weight.dw[0,:],'r')
plt.plot(kx[0,:],weight.d2w[0,:],'g')
#plt.show()

######Play with class Moment#######

mom = momentcalc.Moment()
#print mom.NE
#print mom.even
#print mom.odd
momrot = mom.rotate(np.pi/2.)
momflip= mom.yflip()


######Test class MomentCalculator#######
## xy offsts, even and odd moments 
## from older version of momentcalc for this specific galaxy
dxdy_o=[-0.63159630745,-0.471026751078]
even_o=[3716.1189154, 1579.56184673, 130.113556296, 77.0219184641, 1.0]
odd_o =[-1.13686837722e-13, 4.97379915032e-14]

## now get moments with new version of momentcalc for this specific galaxy
moments = momentcalc.MomentCalculator(kimage_arr,kx,ky,d2k,conjugate,weight,kvar=kvar,id=0)
dx = moments.recenter()
shiftedmoments=moments.get_moment(0,0)
if (testmoments == True):
    print("     ")
    print("Test Moments for Same Galaxy")
    print("Are dx and dy the same?")
    print("dx_new/dx_old = %s" %(dx[0]/dxdy_o[0]))
    print("dy_new/dy_old = %s" %(dx[1]/dxdy_o[1]))
    print("YES!")
    print("          ")
    print("Are even moments the same?")
    print("MF_new/MF_old = %s" %(shiftedmoments.even[0]/even_o[0]))
    print("MR_new/MR_old = %s" %(shiftedmoments.even[1]/even_o[1]))
    print("M1_new/M1_old = %s" %(shiftedmoments.even[2]/even_o[2]))
    print("M2_new/M2_old = %s" %(shiftedmoments.even[3]/even_o[3]))
    print("YES!")

#####Test Covariance Matrix calculation#######
##computed covariance matrix for same galaxy with sigma = 10 (var = 100)
##in old version of momentcalc. rearranged to be in same format as here:
cov_even_o = np.array([[ 1.10064852e+04,  6.93603909e+03,  1.13686838e-13, -6.84548734e-14, 1.0], [ 6.93603909e+03,  6.95762281e+03,  1.98951966e-13, -1.10737569e-13, 1.0], [ 1.13686838e-13,  8.52651283e-14,  3.47939283e+03,  9.50499652e-15, 1.0], [-6.84548734e-14, -1.10737569e-13, -4.02399339e-14,  3.47822998e+03, 1.0]])

cov_odd_o = [[3.46801955e+03,  -4.71014280e-14], [-4.48690557e-14,   3.46801955e+03]]


##need to have noise, so using kimage_arr_wnoise, kvar_wnoise
moments_wnoise = momentcalc.MomentCalculator(kimage_arr_wnoise,kx_wnoise,ky_wnoise,d2k_wnoise,conjugate_wnoise,weight,kvar=kvar_wnoise,id=0)
dx = moments_wnoise.recenter()
shiftedmoments=moments_wnoise.get_moment(0,0)
covariance = moments_wnoise.get_covariance()
if (testcovariancematrix == True):
    print("     ")
    print("Test Covariance Matrices")
    print("Are old and new covariance matrixes for even moments the same?")
    print("MF x [MF, MR, M1, M2] = %s" %(covariance[0][0][0:4]/cov_even_o[0][0:4]))
    print("MR x [MF, MR, M1, M2] = %s" %(covariance[0][1][0:4]/cov_even_o[1][0:4]))
    print("M1 x [MF, MR, M1, M2] = %s" %(covariance[0][2][0:4]/cov_even_o[2][0:4]))
    print("M2 x [MF, MR, M1, M2] = %s" %(covariance[0][3][0:4]/cov_even_o[3][0:4]))
    print("diagonal elements same. M1 and M2 cross other terms have some differences but these are all very small numbers overall")

    print("     ")
    print("Are old and new covariance matrixes for odd moments the same?")
    print("MX x [MX, MY] = %s" %(covariance[1][0]/cov_odd_o[0]))
    print("MY x [MX, MY] = %s" %(covariance[1][1]/cov_odd_o[1]))
    print("diagonal elements same. MX and MY cross each other have some differences but these are all very small numbers overall")

####TEST moment derivatives after shear applied#####

## set up the original galaxy and the sheared galaxy
galaxy = galsimgen.GalaxyGenerator(gal_ellip_sigma, flux_range,hlr_range,noise_var,g1,g2,eseed,gseed)
gal1 = galaxy.sample()
image_arr1 = galsimgen.return_array(gal1,pixel_scale,stamp_size,noise_level = galaxy.noise_var, use_gaussian_noise = True,convolve_with_psf = True, psf = psf)
kimage_arr1, kx1, ky1, d2k1, conjugate1, kvar1 = momentcalc.simpleImage(image_arr1,24,24,psf_arr,pixel_scale=pixel_scale,pixel_noise=np.sqrt(noise_var))

# now shear that same galaxy by some amount
g1shear = 0.001
g2shear = 0.00
galaxy = galsimgen.GalaxyGenerator(gal_ellip_sigma, flux_range,hlr_range,noise_var,g1shear,g2shear,eseed,gseed)
gal2 = galaxy.sample()
image_arr2 = galsimgen.return_array(gal2,pixel_scale,stamp_size,noise_level = galaxy.noise_var, use_gaussian_noise = True,convolve_with_psf = True, psf = psf)
kimage_arr2, kx2, ky2, d2k2, conjugate2, kvar2 = momentcalc.simpleImage(image_arr2,24,24,psf_arr,pixel_scale=pixel_scale,pixel_noise=np.sqrt(noise_var))

# now shear that same galaxy in opposite direction
g1shearb = -1.0*g1shear
g2shearb = -1.0*g2shear
galaxy = galsimgen.GalaxyGenerator(gal_ellip_sigma, flux_range,hlr_range,noise_var,g1shearb,g2shearb,eseed,gseed)
gal2b = galaxy.sample()
image_arr2b = galsimgen.return_array(gal2b,pixel_scale,stamp_size,noise_level = galaxy.noise_var, use_gaussian_noise = True,convolve_with_psf = True, psf = psf)
kimage_arr2b, kx2b, ky2b, d2k2b, conjugate2b, kvar2b = momentcalc.simpleImage(image_arr2b,24,24,psf_arr,pixel_scale=pixel_scale,pixel_noise=np.sqrt(noise_var))

## compute moments of non-sheared galaxy
moments1 = momentcalc.MomentCalculator(kimage_arr1,kx1,ky1,d2k1,conjugate1,weight,kvar=kvar1,id=0)
if recenter:
    dx1 = moments1.recenter()
shiftedmoments1=moments1.get_moment(0,0)
template1=moments1.get_template(0,0)
m1 = template1.get_moment()
m1_dmu = template1.get_dmu()
m1_dg1 = template1.get_dg1()
m1_dg2 = template1.get_dg2()

m1_dg1_old = [419.75417807, 191.32373685, -681.74241831, 34.10978097, 3.75888407, 0.05245587]
m1_dg2_old = [248.52760922, 113.43453703, 34.10978097, -719.17188049, -1.40513787, 6.59839325]

## compare with original moments:
#print("       ")
#print("Compare moments computed two different places")
#print("MF: %s = %s" %(shiftedmoments1.even[m.M0],m1_0.even[m.M0]))
#print("MR: %s = %s" %(shiftedmoments1.even[m.MR],m1_0.even[m.MR]))
#print("M1: %s = %s" %(shiftedmoments1.even[m.M1],m1_0.even[m.M1]))
#print("M2: %s = %s" %(shiftedmoments1.even[m.M2],m1_0.even[m.M2]))
#print("MC: %s = %s" %(shiftedmoments1.even[m.MC],m1_0.even[m.MC]))
#print("MX: %s = %s" %(shiftedmoments1.odd[m.MX],m1_0.odd[m.MX]))
#print("MY: %s = %s" %(shiftedmoments1.odd[m.MY],m1_0.odd[m.MY]))
#print("looks good!")
if ((test1stderivwrtshear == True) & (compwitholdvals == True)):
    print("    ")
    print("Test 1st Derivatives")
    print(" NEW Code 1st derivatives of Moments with respect to g1 and g2")
    print("dMF/dg1 , dMF/dg2 = %s , %s" %(m1_dg1.even[m.M0],m1_dg2.even[m.M0]))
    print("dMR/dg1 , dMR/dg2 = %s , %s" %(m1_dg1.even[m.MR],m1_dg2.even[m.MR]))
    print("dM1/dg1 , dM1/dg2 = %s , %s" %(m1_dg1.even[m.M1],m1_dg2.even[m.M1]))
    print("dM2/dg1 , dM2/dg2 = %s , %s" %(m1_dg1.even[m.M2],m1_dg2.even[m.M2]))
    print("dMC/dg1 , dM2/dg2 = %s , %s" %(m1_dg1.even[m.MC],m1_dg2.even[m.MC]))
    print("dMX/dg1 , dMX/dg2 = %s , %s" %(m1_dg1.odd[m.MX],m1_dg2.odd[m.MX]))
    print("dMY/dg1 , dMY/dg2 = %s , %s" %(m1_dg1.odd[m.MY],m1_dg2.odd[m.MY]))

    print("    ")
    print(" OLD Code 1st derivatives of Moments with respect to g1 and g2")
    print("dMF/dg1 , dMF/dg2 = %s , %s" %(m1_dg1_old[0],m1_dg2_old[0]))
    print("dMR/dg1 , dMR/dg2 = %s , %s" %(m1_dg1_old[1],m1_dg2_old[1]))
    print("dM1/dg1 , dM1/dg2 = %s , %s" %(m1_dg1_old[2],m1_dg2_old[2]))
    print("dM2/dg1 , dM2/dg2 = %s , %s" %(m1_dg1_old[3],m1_dg2_old[3]))
    print("dMX/dg1 , dMX/dg2 = %s , %s" %(m1_dg1_old[4],m1_dg2_old[4]))
    print("dMY/dg1 , dMY/dg2 = %s , %s" %(m1_dg1_old[5],m1_dg2_old[5]))

    print("The 1st Derivatives Agree!")

## compute moments of sheared galaxies
moments2 = momentcalc.MomentCalculator(kimage_arr2,kx2,ky2,d2k2,conjugate2,weight,kvar=kvar2,id=0)
if recenter:
    dx2 = moments2.recenter()
shiftedmoments2=moments2.get_moment(0,0)
template2=moments2.get_template(0,0)
m2 = template2.get_moment()
m2_dg1 = template2.get_dg1()
m2_dg2 = template2.get_dg2()

moments2b = momentcalc.MomentCalculator(kimage_arr2b,kx2b,ky2b,d2k2b,conjugate2b,weight,kvar=kvar2b,id=0)
if recenter:
    dx2b = moments2b.recenter()
shiftedmoments2b=moments2b.get_moment(0,0)
template2b=moments2b.get_template(0,0)
m2b = template2b.get_moment()
m2b_dg1 = template2b.get_dg1()
m2b_dg2 = template2b.get_dg2()

if ((test1stderivwrtshear == True) & (compwitholdvals == True)):

    print("      ")
    print("Test whether recover the new moment after applying a small shear to the galaxy")
    print("Applying g1 = %s and g2 = %s" %(g1shear, g2shear))
    print("orig MF = %s" %(shiftedmoments1.even[m.M0]))
    print("expected MF from 1st derivs (new) = %s" %(shiftedmoments1.even[m.M0] + m1_dg1.even[m.M0]*g1shear + m1_dg2.even[m.M0]*g2shear))
    print("expected MF from 1st derivs (old) = %s" %(shiftedmoments1.even[m.M0] + m1_dg1_old[0]*g1shear + m1_dg2_old[0]*g2shear))
    print("calculated MF after shear applied = %s" %(shiftedmoments2.even[m.M0]))
    print("      ")
    print("orig MR = %s" %(shiftedmoments1.even[m.MR]))
    print("expected MR from 1st derivs (new) = %s" %(shiftedmoments1.even[m.MR] + m1_dg1.even[m.MR]*g1shear + m1_dg2.even[m.MR]*g2shear))
    print("expected MR from 1st derivs (old) = %s" %(shiftedmoments1.even[m.MR] + m1_dg1_old[1]*g1shear + m1_dg2_old[1]*g2shear))
    print("calculated MR after shear applied = %s" %(shiftedmoments2.even[m.MR]))
    print("      ")
    print("orig M1 = %s" %(shiftedmoments1.even[m.M1]))
    print("expected M1 from 1st derivs (new) = %s" %(shiftedmoments1.even[m.M1] + m1_dg1.even[m.M1]*g1shear + m1_dg2.even[m.M1]*g2shear))
    print("expected M1 from 1st derivs (old) = %s" %(shiftedmoments1.even[m.M1] + m1_dg1_old[2]*g1shear + m1_dg2_old[2]*g2shear))
    print("calculated MF after shear applied = %s" %(shiftedmoments2.even[m.M1]))
    print("      ")
    print("orig M2 = %s" %(shiftedmoments1.even[m.M2]))
    print("expected M2 from 1st derivs (new) = %s" %(shiftedmoments1.even[m.M2] + m1_dg1.even[m.M2]*g1shear + m1_dg2.even[m.M2]*g2shear))
    print("expected M2 from 1st derivs (old) = %s" %(shiftedmoments1.even[m.M2] + m1_dg1_old[3]*g1shear + m1_dg2_old[3]*g2shear))
    print("calculated MF after shear applied = %s" %(shiftedmoments2.even[m.M2]))

    print("All look Good!")

if ((test1stderivwrtshear == True) & (finitedifftest == True)):
    print("Central Finite Difference Test of 1st Derivatives wrt shear")
    #F(x + dg1 + dg2) = F(x) + dF/dg1 * dg1 + dF/dg2 * dg2
    #F(x - dg1 - dg2) = F(x) - dF/dg1 * dg1 - dF/dg2 * dg2
    #F(x + dg1 + dg2) - F(x-dg1-dg2) = 2dF/dg1 * dg1 + 2dF/dg2 * dg2

    print("[F(x+dg1+dg2) - F(x-dg1-dg2)]/2 = dF/dg1 * dg1 + dF/dg2 * dg2")
    print("MF: %s = %s" %((shiftedmoments2.even[m.M0] - shiftedmoments2b.even[m.M0])/2., m1_dg1.even[m.M0] * g1shear + m1_dg2.even[m.M0] * g2shear))
    print("MR: %s = %s" %((shiftedmoments2.even[m.MR] - shiftedmoments2b.even[m.MR])/2., m1_dg1.even[m.MR] * g1shear + m1_dg2.even[m.MR] * g2shear))
    print("M1: %s = %s" %((shiftedmoments2.even[m.M1] - shiftedmoments2b.even[m.M1])/2., m1_dg1.even[m.M1] * g1shear + m1_dg2.even[m.M1] * g2shear))
    print("M2: %s = %s" %((shiftedmoments2.even[m.M2] - shiftedmoments2b.even[m.M2])/2., m1_dg1.even[m.M2] * g1shear + m1_dg2.even[m.M2] * g2shear))
    print("MC: %s = %s" %((shiftedmoments2.even[m.MC] - shiftedmoments2b.even[m.MC])/2., m1_dg1.even[m.MC] * g1shear + m1_dg2.even[m.MC] * g2shear))
    print("MX: %s = %s" %((shiftedmoments2.odd[m.MX] - shiftedmoments2b.odd[m.MX])/2., m1_dg1.odd[m.MX] * g1shear + m1_dg2.odd[m.MX] * g2shear))
    print("MY: %s = %s" %((shiftedmoments2.odd[m.MY] - shiftedmoments2b.odd[m.MY])/2., m1_dg1.odd[m.MY] * g1shear + m1_dg2.odd[m.MY] * g2shear))
    print("If turn off recentering X & Y derivatives agree with expectations")



#####TEST 2nd Derivatives of Galaxy 1
# first compare with what we have from old code (not vetted, so if don't match, not necessarily a problem


m1_dg1_dg1 = template1.get_dg1_dg1()
m1_dg2_dg2 = template1.get_dg2_dg2()
m1_dg1_dg2 = template1.get_dg1_dg2()
m1_dmu_dmu = template1.get_dmu_dmu()
m1_dmu_dg1 = template1.get_dmu_dg1()
m1_dmu_dg2 = template1.get_dmu_dg2()

m1_dg1_dg1_old = [-2.20158163e+03, -1.01042160e+03, -4.17187022e+02, -1.66541858e+02, 1.45384712e+01, -1.93609234e+00]
m1_dg2_dg2_old = [-2.32117411e+03, -1.06134469e+03, -3.11421568e+02, -2.63873479e+02, 1.48283380e+01, -1.74049800e-01]
m1_dg1_dg2_old = [1.10022419e+02 , 5.00930149e+01, -1.24980209e+01, -5.11944556e+01 , -6.30811438e-01,  8.73234759e-01 ]

if ((test2ndderivwrtshear == True) & (compwitholdvals == True)):
    print("    ")
    print("Test 2nd Derivatives")
    print("Compare New vs. Old 2nd derivative terms")
    print("d2MF/dg1dg1: %s = %s " %(m1_dg1_dg1.even[m.M0], m1_dg1_dg1_old[0]))
    print("d2MF/dg2dg2: %s = %s " %(m1_dg2_dg2.even[m.M0], m1_dg2_dg2_old[0]))
    print("d2MF/dg1dg2: %s = %s " %(m1_dg1_dg2.even[m.M0], m1_dg1_dg2_old[0]))
    print("    ")
    print("d2MR/dg1dg1: %s = %s " %(m1_dg1_dg1.even[m.MR], m1_dg1_dg1_old[1]))
    print("d2MR/dg2dg2: %s = %s " %(m1_dg2_dg2.even[m.MR], m1_dg2_dg2_old[1]))
    print("d2MR/dg1dg2: %s = %s " %(m1_dg1_dg2.even[m.MR], m1_dg1_dg2_old[1]))
    print("    ")
    print("d2M1/dg1dg1: %s = %s " %(m1_dg1_dg1.even[m.M1], m1_dg1_dg1_old[2]))
    print("d2M1/dg2dg2: %s = %s " %(m1_dg2_dg2.even[m.M1], m1_dg2_dg2_old[2]))
    print("d2M1/dg1dg2: %s = %s " %(m1_dg1_dg2.even[m.M1], m1_dg1_dg2_old[2]))
    print("    ")
    print("d2M2/dg1dg1: %s = %s " %(m1_dg1_dg1.even[m.M2], m1_dg1_dg1_old[3]))
    print("d2M2/dg2dg2: %s = %s " %(m1_dg2_dg2.even[m.M2], m1_dg2_dg2_old[3]))
    print("d2M2/dg1dg2: %s = %s " %(m1_dg1_dg2.even[m.M2], m1_dg1_dg2_old[3]))
    print("    ")
    print("d2MX/dg1dg1: %s = %s " %(m1_dg1_dg1.odd[m.MX], m1_dg1_dg1_old[4]))
    print("d2MX/dg2dg2: %s = %s " %(m1_dg2_dg2.odd[m.MX], m1_dg2_dg2_old[4]))
    print("d2MX/dg1dg2: %s = %s " %(m1_dg1_dg2.odd[m.MX], m1_dg1_dg2_old[4]))
    print("    ")
    print("d2MY/dg1dg1: %s = %s " %(m1_dg1_dg1.odd[m.MY], m1_dg1_dg1_old[5]))
    print("d2MY/dg2dg2: %s = %s " %(m1_dg2_dg2.odd[m.MY], m1_dg2_dg2_old[5]))
    print("d2MY/dg1dg2: %s = %s " %(m1_dg1_dg2.odd[m.MY], m1_dg1_dg2_old[5]))
    print(" all agree!")


    print("   ")
    print("Test whether recover the new 1st derivatives after applying a small shear to the galaxy")
    print("Flux and R Moments")
    print("orig dMF/dg1 = %s" %(m1_dg1.even[m.M0]))
    print("updated dMF/dg1 from (new) 2nd derivs = %s " %(m1_dg1.even[m.M0] + m1_dg1_dg1.even[m.M0]*g1shear + m1_dg1_dg2.even[m.M0]*g2shear))
    print("updated dMF/dg1 from (old) 2nd derivs = %s" %(m1_dg1.even[m.M0] + m1_dg1_dg1_old[0]*g1shear + m1_dg1_dg2_old[0]*g2shear))
    print("dMF/dg1 from sheared gal              = %s" %(m2_dg1.even[m.M0]))
    print("   ")
    print("orig dMF/dg2 = %s" %(m1_dg2.even[m.M0]))
    print("updated dMF/dg2 from (new) 2nd derivs = %s " %(m1_dg2.even[m.M0] + m1_dg2_dg2.even[m.M0]*g2shear + m1_dg1_dg2.even[m.M0]*g1shear))
    print("updated dMF/dg2 from (old) 2nd derivs = %s" %(m1_dg2.even[m.M0] + m1_dg2_dg2_old[0]*g2shear + m1_dg1_dg2_old[0]*g1shear))
    print("dMF/dg2 from sheared gal              = %s" %(m2_dg2.even[m.M0]))
    print("    ")
    print("orig dMR/dg1 = %s" %(m1_dg1.even[m.MR]))
    print("updated dMR/dg1 from (new) 2nd derivs = %s " %(m1_dg1.even[m.MR] + m1_dg1_dg1.even[m.MR]*g1shear + m1_dg1_dg2.even[m.MR]*g2shear))
    print("updated dMR/dg1 from (old) 2nd derivs = %s" %(m1_dg1.even[m.MR] + m1_dg1_dg1_old[1]*g1shear + m1_dg1_dg2_old[1]*g2shear))
    print("dMR/dg1 from sheared gal              = %s" %(m2_dg1.even[m.MR]))
    print("   ")
    print("orig dMR/dg2 = %s" %(m1_dg2.even[m.MR]))
    print("updated dMR/dg2 from (new) 2nd derivs = %s " %(m1_dg2.even[m.MR] + m1_dg2_dg2.even[m.MR]*g2shear + m1_dg1_dg2.even[m.MR]*g1shear))
    print("updated dMR/dg2 from (old) 2nd derivs = %s" %(m1_dg2.even[m.MR] + m1_dg2_dg2_old[1]*g2shear + m1_dg1_dg2_old[1]*g1shear))
    print("dMR/dg2 from sheared gal              = %s" %(m2_dg2.even[m.MR]))
    print("Flux and R moments look good!")
    print("    ")
    print("E moments")
    print("orig dM1/dg1 = %s" %(m1_dg1.even[m.M1]))
    print("updated dM1/dg1 from (new) 2nd derivs = %s " %(m1_dg1.even[m.M1] + m1_dg1_dg1.even[m.M1]*g1shear + m1_dg1_dg2.even[m.M1]*g2shear))
    print("updated dM1/dg1 from (old) 2nd derivs = %s" %(m1_dg1.even[m.M1] + m1_dg1_dg1_old[2]*g1shear + m1_dg1_dg2_old[2]*g2shear))
    print("dM1/dg1 from sheared gal              = %s" %(m2_dg1.even[m.M1]))
    print("   ")
    print("orig dM1/dg2 = %s" %(m1_dg2.even[m.M1]))
    print("updated dM1/dg2 from (new) 2nd derivs = %s " %(m1_dg2.even[m.M1] + m1_dg2_dg2.even[m.M1]*g2shear + m1_dg1_dg2.even[m.M1]*g1shear))
    print("updated dM1/dg2 from (old) 2nd derivs = %s" %(m1_dg2.even[m.M1] + m1_dg2_dg2_old[2]*g2shear + m1_dg1_dg2_old[2]*g1shear))
    print("dM1/dg2 from sheared gal              = %s" %(m2_dg2.even[m.M1]))
    print("    ")
    print("orig dM2/dg1 = %s" %(m1_dg1.even[m.M2]))
    print("updated dM2/dg1 from (new) 2nd derivs = %s" %(m1_dg1.even[m.M2] + m1_dg1_dg1.even[m.M2]*g1shear + m1_dg1_dg2.even[m.M2]*g2shear))
    print("updated dM2/dg1 from (old) 2nd derivs = %s" %(m1_dg1.even[m.M2] + m1_dg1_dg1_old[3]*g1shear + m1_dg1_dg2_old[3]*g2shear))
    print("dM2/dg1 from sheared gal              = %s" %(m2_dg1.even[m.M2]))
    print("   ")
    print("orig dM2/dg2 = %s" %(m1_dg2.even[m.M2]))
    print("updated dM2/dg2 from (new) 2nd derivs = %s " %(m1_dg2.even[m.M2] + m1_dg2_dg2.even[m.M2]*g2shear + m1_dg1_dg2.even[m.M2]*g1shear))
    print("updated dM2/dg2 from (old) 2nd derivs = %s" %(m1_dg2.even[m.M2] + m1_dg2_dg2_old[3]*g2shear + m1_dg1_dg2_old[3]*g1shear))
    print("dM2/dg2 from sheared gal              = %s" %(m2_dg2.even[m.M2]))
    print(" Looks Good!")
    print("    ")
    print(" XY moments  ")
    print("orig dMX/dg1 = %s" %(m1_dg1.odd[m.MX]))
    print("updated dMX/dg1 from (new) 2nd derivs = %s " %(m1_dg1.odd[m.MX] + m1_dg1_dg1.odd[m.MX]*g1shear + m1_dg1_dg2.odd[m.MX]*g2shear))
    print("updated dMX/dg1 from (old) 2nd derivs = %s" %(m1_dg1.odd[m.MX] + m1_dg1_dg1_old[4]*g1shear + m1_dg1_dg2_old[4]*g2shear))
    print("dMX/dg1 from sheared gal              = %s" %(m2_dg1.odd[m.MX]))
    print("   ")
    print("orig dMX/dg2 = %s" %(m1_dg2.odd[m.MX]))
    print("updated dMX/dg2 from (new) 2nd derivs = %s " %(m1_dg2.odd[m.MX] + m1_dg2_dg2.odd[m.MX]*g2shear + m1_dg1_dg2.odd[m.MX]*g1shear))
    print("updated dMX/dg2 from (old) 2nd derivs = %s" %(m1_dg2.odd[m.MX] + m1_dg2_dg2_old[4]*g2shear + m1_dg1_dg2_old[4]*g1shear))
    print("dMX/dg2 from sheared gal              = %s" %(m2_dg2.odd[m.MX]))
    print("    ")
    print("orig dMY/dg1 = %s" %(m1_dg1.odd[m.MY]))
    print("updated dMY/dg1 from (new) 2nd derivs = %s " %(m1_dg1.odd[m.MY] + m1_dg1_dg1.odd[m.MY]*g1shear + m1_dg1_dg2.odd[m.MY]*g2shear))
    print("updated dMY/dg1 from (old) 2nd derivs = %s" %(m1_dg1.odd[m.MY] + m1_dg1_dg1_old[5]*g1shear + m1_dg1_dg2_old[5]*g2shear))
    print("dMY/dg1 from sheared gal              = %s" %(m2_dg1.odd[m.MY]))
    print("   ")
    print("orig dMY/dg2 = %s" %(m1_dg2.odd[m.MY]))
    print("updated dMY/dg2 from (new) 2nd derivs = %s " %(m1_dg2.odd[m.MY] + m1_dg2_dg2.odd[m.MY]*g2shear + m1_dg1_dg2.odd[m.MY]*g1shear))
    print("updated dMY/dg2 from (old) 2nd derivs = %s" %(m1_dg2.odd[m.MY] + m1_dg2_dg2_old[5]*g2shear + m1_dg1_dg2_old[5]*g1shear))
    print("dMY/dg2 from sheared gal              = %s" %(m2_dg2.odd[m.MY]))
    print("For MX and MY, new and old code give same second derivatives, but don't quite agree with what the shear pattern says - not sure whether an issue?")
    print("    ")

if ((test2ndderivwrtshear == True) & (finitedifftest == True)):
    print("   ")
    print("Central Finite Difference Test of 2nd Derivatives wrt shear")
    print("F(x+dg1+dg2)-2F(x)+F(x-dg1-dg2) = d2F/dg1dg1*g1shear^2 + d2F/dg2/dg2*g2shear^2 + 2*d2F/dg1dg2*g1shear*g2shear")
    print("MF: %s = %s" %(shiftedmoments2.even[m.M0] -2*shiftedmoments1.even[m.M0] + shiftedmoments2b.even[m.M0], m1_dg1_dg1.even[m.M0] * g1shear**2 + m1_dg2_dg2.even[m.M0] * g2shear**2 + 2.0*m1_dg1_dg2.even[m.M0] * g1shear*g2shear))
    print("MR: %s = %s" %(shiftedmoments2.even[m.MR] -2*shiftedmoments1.even[m.MR] + shiftedmoments2b.even[m.MR], m1_dg1_dg1.even[m.MR] * g1shear**2 + m1_dg2_dg2.even[m.MR] * g2shear**2 + 2.0*m1_dg1_dg2.even[m.MR] * g1shear*g2shear))
    print("M1: %s = %s" %(shiftedmoments2.even[m.M1] -2*shiftedmoments1.even[m.M1] + shiftedmoments2b.even[m.M1], m1_dg1_dg1.even[m.M1] * g1shear**2 + m1_dg2_dg2.even[m.M1] * g2shear**2 + 2.0*m1_dg1_dg2.even[m.M1] * g1shear*g2shear))
    print("M2: %s = %s" %(shiftedmoments2.even[m.M2] -2*shiftedmoments1.even[m.M2] + shiftedmoments2b.even[m.M2], m1_dg1_dg1.even[m.M2] * g1shear**2 + m1_dg2_dg2.even[m.M2] * g2shear**2 + 2.0*m1_dg1_dg2.even[m.M2] * g1shear*g2shear))
    print("MC: %s = %s" %(shiftedmoments2.even[m.MC] -2*shiftedmoments1.even[m.MC] + shiftedmoments2b.even[m.MC], m1_dg1_dg1.even[m.MC] * g1shear**2 + m1_dg2_dg2.even[m.MC] * g2shear**2 + 2.0*m1_dg1_dg2.even[m.MC] * g1shear*g2shear))
    print("MX: %s = %s" %(shiftedmoments2.odd[m.MX] -2*shiftedmoments1.odd[m.MX] + shiftedmoments2b.odd[m.MX], m1_dg1_dg1.odd[m.MX] * g1shear**2 + m1_dg2_dg2.odd[m.MX] * g2shear**2 + 2.0*m1_dg1_dg2.odd[m.MX] * g1shear*g2shear))
    print("MY: %s = %s" %(shiftedmoments2.odd[m.MY] -2*shiftedmoments1.odd[m.MY] + shiftedmoments2b.odd[m.MY], m1_dg1_dg1.odd[m.MY] * g1shear**2 + m1_dg2_dg2.odd[m.MY] * g2shear**2 + 2.0*m1_dg1_dg2.odd[m.MY] * g1shear*g2shear))


testmc=False
######TEST concentration moment on Gaussian Galaxy#####
if (testmc):
    import galsim
    # make galsim objects
    gausspsf = galsim.Gaussian(fwhm=3.0)
    gaussgal = galsim.Gaussian(fwhm=5.0,flux=1000.0)
    gaussfinal = galsim.Convolve([gaussgal,gausspsf])
    newsigma = np.sqrt(gausspsf.sigma**2 + gaussgal.sigma**2)
    # draw iages
    gausspsfarr = gausspsf.drawImage(nx=48,ny=48,scale=1.0,use_true_center=False).array
    gaussgalarr = gaussgal.drawImage(nx=48,ny=48,scale=1.0,use_true_center=False).array
    gaussfinarr = gaussfinal.drawImage(nx=48,ny=48,scale=1.0,use_true_center=False).array
    # make k-images
    gausspsfkarr = np.fft.fftshift(np.fft.fft2(gausspsfarr))
    gaussgalkarr = np.fft.fftshift(np.fft.fft2(gaussgalarr))
    gaussfinkarr = np.fft.fftshift(np.fft.fft2(gaussfinarr))

    gkimage_arr, gkx, gky, gd2k, gconjugate, gkvar = momentcalc.simpleImage(gaussfinarr,24,24,gausspsfarr,pixel_scale=pixel_scale,pixel_noise=0.0)
    gaussmoments = momentcalc.MomentCalculator(gkimage_arr,gkx,gky,gd2k,gconjugate,weight,kvar=kvar,id=0)
    gdx2 = gaussmoments.recenter()
    gaussshiftedmoments=gaussmoments.get_moment(0,0)

    gkk = np.fft.fftshift(np.fft.fftfreq(48))*2.0*np.pi
    gkkx = np.tile(gkk,(48,1))
    gkky=gkkx.transpose()
    gdk = gkk[1]-gkk[0]
    gsigw=1.2
    gausswkarr = np.exp(-(gkkx**2+gkky**2)*gsigw**2)
    gausswkarrp = -2.0 * gsigw**2 * (gkkx + gkky) * gausswkarr
    galktest = 1000.*np.exp(-(gkkx**2+gkky**2)*(newsigma/np.sqrt(2))**2)
 
    gsigk = np.sqrt((newsigma/np.sqrt(2.0))**2 + gsigw**2 - (gausspsf.sigma/np.sqrt(2.0))**2)
    Ak = gaussfinkarr[24,24].real
    Bk = gausspsfkarr[24,24].real
    Ck = gausswkarr[24,24]
    k = gkkx + 1j * gkky
    kbar = gkkx - 1j * gkky
    Fk = k**2 * kbar**2
    gam3_2=0.886227
    gam5_2=1.32934
    print("   ")
    print("Test Concentration Moment")
    print("Pen & Paper Gaussian = %s" %( (Ak * Ck * 2.0 / Bk) * (np.sqrt(np.pi/gsigk**2) * gam5_2/gsigk**5 + (gam3_2/gsigk**3)**2))) 
    print("numerical Gaussian = %s" %(np.sum(gaussfinkarr * gausswkarr *gdk**2 *Fk / gausspsfkarr)))
    print("from momentcalc = %s" %(gaussshiftedmoments.even[gaussshiftedmoments.MC]))


    #plt.figure(2)
    #plt.plot(gkk,np.abs(gausspsfkarr[24,:]),'b')
    #plt.plot(gkk,np.abs(gaussfinkarr[24,:])/1000.,'g')
    #plt.plot(gkk,np.abs(galktest[24,:])/1000.,'r')
 #   plt.show()
    pdb.set_trace()
#    MC_expect = 



####TEST moment derivatives after magnification applied#####
# now magnify the same galaxy by some amount
g1shear = 0.00
g2shear = 0.00
mu =  (1.0 + 0.02)
mub = (1.0 - 0.02)
galaxy = galsimgen.GalaxyGenerator(gal_ellip_sigma, flux_range,hlr_range,noise_var,g1shear,g2shear,eseed,gseed)
gal3 = galaxy.sample()
gal3b = gal3
# can use expand (linear rescaling, preserves surface brightness), dilate (linear rescaling, preserves flux), magnify (area rescaling (factor of sqrt(mu), preserves surfacebrightness)
gal3 = gal3.magnify(mu)
gal3b = gal3b.magnify(mub)

if True:
    # Gary test
    g0 = galsimgen.GalaxyGenerator(gal_ellip_sigma, flux_range,hlr_range,noise_var,g1shear,g2shear,eseed,gseed).sample()
    dmu = 0.02
    gp = g0.magnify( (1+dmu)**2)
    gm = g0.magnify( (1-dmu)**2)
    print "Gary's galaxies' fluxes test (should be equal):",g0.getFlux(),gp.getFlux()/(1+dmu)**2,gm.getFlux()/(1-dmu)**2
    
    def makeMC(gal):
        # Return a MomentCalculator from a galsim galaxy
        img = galsimgen.return_array(gal,pixel_scale,stamp_size,noise_level = 0.,
                                     use_gaussian_noise = True,convolve_with_psf = True, psf = psf)
        kimg, kx, ky, d2k, conj, kvar = momentcalc.simpleImage(img,24,24,psf_arr,
                                                                   pixel_scale=pixel_scale,pixel_noise=0.)
        return momentcalc.MomentCalculator(kimg,kx,ky,d2k,conj,weight,kvar=kvar)
    mc0 = makeMC(g0)
    m0 = mc0.get_moment(0.,0.).even
    mp = makeMC(gp).get_moment(0.,0.).even
    mm = makeMC(gm).get_moment(0.,0.).even
    print "Numerical:", (mp-2*m0+mm)/(dmu*dmu)
    print "Analytic:", mc0.get_template(0.,0.).get_dmu_dmu().even

    
image_arr3 = galsimgen.return_array(gal3,pixel_scale,stamp_size,noise_level = galaxy.noise_var, use_gaussian_noise = True,convolve_with_psf = True, psf = psf)
kimage_arr3, kx3, ky3, d2k3, conjugate3, kvar3 = momentcalc.simpleImage(image_arr3,24,24,psf_arr,pixel_scale=pixel_scale,pixel_noise=np.sqrt(noise_var))

image_arr3b = galsimgen.return_array(gal3b,pixel_scale,stamp_size,noise_level = galaxy.noise_var, use_gaussian_noise = True,convolve_with_psf = True, psf = psf)
kimage_arr3b, kx3b, ky3b, d2k3b, conjugate3b, kvar3b = momentcalc.simpleImage(image_arr3b,24,24,psf_arr,pixel_scale=pixel_scale,pixel_noise=np.sqrt(noise_var))

## compute moments of magnified galaxy
moments3 = momentcalc.MomentCalculator(kimage_arr3,kx3,ky3,d2k3,conjugate3,weight,kvar=kvar3,id=0)
if recenter:
    dx3 = moments3.recenter()
shiftedmoments3=moments3.get_moment(0,0)
template3=moments3.get_template(0,0)
m3 = template3.get_moment()
m3_dmu = template3.get_dmu()

moments3b = momentcalc.MomentCalculator(kimage_arr3b,kx3b,ky3b,d2k3b,conjugate3b,weight,kvar=kvar3b,id=0)
if recenter:
    dx3b = moments3b.recenter()
shiftedmoments3b=moments3b.get_moment(0,0)
if ((test1stderivwrtmag == True) & (finitedifftest == False)):
    dmu = mu - 1.0
    print("      ")
    print("TEST magnification 1st derivatives")
    print("Test whether recover the new moment after applying a small magnification to the galaxy")
    print("Applying mu = %s" %(mu))
    print("orig MF = %s" %(shiftedmoments1.even[m.M0]))
    print("expected MF from 1st derivs (new) = %s" %(shiftedmoments1.even[m.M0] + m1_dmu.even[m.M0]*dmu))
    print("calculated MF after magnification applied = %s" %(shiftedmoments3.even[m.M0]))
    print("      ")
    print("orig MR = %s" %(shiftedmoments1.even[m.MR]))
    print("expected MR from 1st derivs (new) = %s" %(shiftedmoments1.even[m.MR] + m1_dmu.even[m.MR]*dmu))
    print("calculated MR after magnification applied = %s" %(shiftedmoments3.even[m.MR]))
    print("      ")
    print("orig M1 = %s" %(shiftedmoments1.even[m.M1]))
    print("expected M1 from 1st derivs (new) = %s" %(shiftedmoments1.even[m.M1] + m1_dmu.even[m.M1]*dmu))
    print("calculated M1 after magnification applied = %s" %(shiftedmoments3.even[m.M1]))
    print("      ")
    print("orig M2 = %s" %(shiftedmoments1.even[m.M2]))
    print("expected M2 from 1st derivs (new) = %s" %(shiftedmoments1.even[m.M2] + m1_dmu.even[m.M2]*dmu))
    print("calculated M2 after magnification applied = %s" %(shiftedmoments3.even[m.M2]))
    print("      ")
    print("orig MX = %s" %(shiftedmoments1.odd[m.MX]))
    print("expected MX from 1st derivs (new) = %s" %(shiftedmoments1.odd[m.MX] + m1_dmu.odd[m.MX]*dmu))
    print("calculated MX after magnification applied = %s" %(shiftedmoments3.odd[m.MX]))
    print("      ")
    print("orig MY = %s" %(shiftedmoments1.odd[m.MY]))
    print("expected MY from 1st derivs (new) = %s" %(shiftedmoments1.odd[m.MY] + m1_dmu.odd[m.MY]*dmu))
    print("calculated MY after magnification applied = %s" %(shiftedmoments3.odd[m.MY]))
    print("all agree except MX and MY, not sure whether those matter?")
    print("      ")

if ((test1stderivwrtmag == True) & (finitedifftest == True)):
    print("   ")
    print("Central Finite Difference Test of 1st Derivatives wrt mu")
    print("[F(x+dmu) - F(x-dmu)]/2 = dF/dmu * dmu")
    dmu = mu-1.0
    print("MF: %s = %s" %((shiftedmoments3.even[m.M0] - shiftedmoments3b.even[m.M0])/2., m1_dmu.even[m.M0] * dmu))
    print("MR: %s = %s" %((shiftedmoments3.even[m.MR] - shiftedmoments3b.even[m.MR])/2., m1_dmu.even[m.MR] * dmu))
    print("M1: %s = %s" %((shiftedmoments3.even[m.M1] - shiftedmoments3b.even[m.M1])/2., m1_dmu.even[m.M1] * dmu))
    print("M2: %s = %s" %((shiftedmoments3.even[m.M2] - shiftedmoments3b.even[m.M2])/2., m1_dmu.even[m.M2] * dmu))
    print("MC: %s = %s" %((shiftedmoments3.even[m.MC] - shiftedmoments3b.even[m.MC])/2., m1_dmu.even[m.MC] * dmu))
    print("MX: %s = %s" %((shiftedmoments3.odd[m.MX] - shiftedmoments3b.odd[m.MX])/2., m1_dmu.odd[m.MX] * dmu))
    print("MY: %s = %s" %((shiftedmoments3.odd[m.MY] - shiftedmoments3b.odd[m.MY])/2., m1_dmu.odd[m.MY] * dmu))

    print("All Look Good!")

if ((test2ndderivwrtmag == True) & (finitedifftest == False)):
    dmu = mu - 1.0
    print("orig dMF/dmu = %s" %(m1_dmu.even[m.M0]))
    print("updated dMF/dmu from 2nd deriv = %s " %(m1_dmu.even[m.M0] + m1_dmu_dmu.even[m.M0]*(np.sqrt(mu)-1.0)))
    print("dMF/dmu from magnified gal     = %s" %(m3_dmu.even[m.M0]))
    print("   ")

if ((test2ndderivwrtmag == True) & (finitedifftest == True)):

    print("   ")
    print("Central Finite Difference Test of 2nd Derivatives wrt mu")
    print("[F(x+dmu) -2F(x) + F(x-dmu)] = d2F/dmudmu * dmu^2")
    dmu = mu - 1.0
    print("MF: %s = %s" %((shiftedmoments3.even[m.M0] - 2.0*shiftedmoments1.even[m.M0] + shiftedmoments3b.even[m.M0]), m1_dmu_dmu.even[m.M0]*dmu**2))
    print("MR: %s = %s" %((shiftedmoments3.even[m.MR] - 2.0*shiftedmoments1.even[m.MR] + shiftedmoments3b.even[m.MR]), m1_dmu_dmu.even[m.MR]*dmu**2))
    print("M1: %s = %s" %((shiftedmoments3.even[m.M1] - 2.0*shiftedmoments1.even[m.M1] + shiftedmoments3b.even[m.M1]), m1_dmu_dmu.even[m.M1]*dmu**2))
    print("M2: %s = %s" %((shiftedmoments3.even[m.M2] - 2.0*shiftedmoments1.even[m.M2] + shiftedmoments3b.even[m.M2]), m1_dmu_dmu.even[m.M2]*dmu**2))
    print("MC: %s = %s" %((shiftedmoments3.even[m.MC] - 2.0*shiftedmoments1.even[m.MC] + shiftedmoments3b.even[m.MC]), m1_dmu_dmu.even[m.MC]*dmu**2))
    print("Looks Good Now!")


print("       ")

##first need to magnify gal2 and gal2b
gal4 = gal2.magnify(mu**2) # + dg1, + dg2 + dmu
gal4b= gal2b.magnify(mub**2) # - dg1 - dg2 -dmu

image_arr4 = galsimgen.return_array(gal4,pixel_scale,stamp_size,noise_level = galaxy.noise_var, use_gaussian_noise = True,convolve_with_psf = True, psf = psf)
kimage_arr4, kx4, ky4, d2k4, conjugate4, kvar4 = momentcalc.simpleImage(image_arr4,24,24,psf_arr,pixel_scale=pixel_scale,pixel_noise=np.sqrt(noise_var))

image_arr4b = galsimgen.return_array(gal4b,pixel_scale,stamp_size,noise_level = galaxy.noise_var, use_gaussian_noise = True,convolve_with_psf = True, psf = psf)
kimage_arr4b, kx4b, ky4b, d2k4b, conjugate4b, kvar4b = momentcalc.simpleImage(image_arr4b,24,24,psf_arr,pixel_scale=pixel_scale,pixel_noise=np.sqrt(noise_var))

## compute moments of magnified and sheared galaxy
moments4 = momentcalc.MomentCalculator(kimage_arr4,kx4,ky4,d2k4,conjugate4,weight,kvar=kvar4,id=0)
if recenter:
    dx4 = moments4.recenter()
shiftedmoments4=moments4.get_moment(0,0)
moments4b = momentcalc.MomentCalculator(kimage_arr4b,kx4b,ky4b,d2k4b,conjugate4b,weight,kvar=kvar4b,id=0)
if recenter:
    dx4b = moments4b.recenter()
shiftedmoments4b=moments4b.get_moment(0,0)
if (test2ndderivwrtshearandmag == True):
    dmu=mu-1.0
    print("Central Finite Difference Test of 2nd Derivatives wrt shear and mag")
    print("F(x+dg1+dg2+dmu)-2F(x)+F(x-dg1-dg2-dmu) = d2F/dg1dg1*g1shear^2 + d2F/dg2/dg2*g2shear^2 + 2*d2F/dg1dg2*g1shear*g2shear + d2F/dmudg1*dmu*g1shear + d2F/dmu/dg2*dmu*g2shear + d2F/dmudmu*dmu^2")
    print("MF: %s = %s" %(shiftedmoments4.even[m.M0] -2*shiftedmoments1.even[m.M0] + shiftedmoments4b.even[m.M0], m1_dg1_dg1.even[m.M0] * g1shear**2 + m1_dg2_dg2.even[m.M0] * g2shear**2 + 2.0*m1_dg1_dg2.even[m.M0] * g1shear*g2shear + m1_dmu_dg1.even[m.M0] * g1shear * dmu + m1_dmu_dg2.even[m.M0]*g2shear*dmu + m1_dmu_dmu.even[m.M0]*dmu**2))
    print("MR: %s = %s" %(shiftedmoments4.even[m.MR] -2*shiftedmoments1.even[m.MR] + shiftedmoments4b.even[m.MR], m1_dg1_dg1.even[m.MR] * g1shear**2 + m1_dg2_dg2.even[m.MR] * g2shear**2 + 2.0*m1_dg1_dg2.even[m.MR] * g1shear*g2shear + m1_dmu_dg1.even[m.MR] * g1shear * dmu + m1_dmu_dg2.even[m.MR]*g2shear*dmu + m1_dmu_dmu.even[m.MR]*dmu**2))
    print("M1: %s = %s" %(shiftedmoments4.even[m.M1] -2*shiftedmoments1.even[m.M1] + shiftedmoments4b.even[m.M1], m1_dg1_dg1.even[m.M1] * g1shear**2 + m1_dg2_dg2.even[m.M1] * g2shear**2 + 2.0*m1_dg1_dg2.even[m.M1] * g1shear*g2shear + m1_dmu_dg1.even[m.M1] * g1shear * dmu + m1_dmu_dg2.even[m.M1]*g2shear*dmu + m1_dmu_dmu.even[m.M1]*dmu**2))
    print("M2: %s = %s" %(shiftedmoments4.even[m.M2] -2*shiftedmoments1.even[m.M2] + shiftedmoments4b.even[m.M2], m1_dg1_dg1.even[m.M2] * g1shear**2 + m1_dg2_dg2.even[m.M2] * g2shear**2 + 2.0*m1_dg1_dg2.even[m.M2] * g1shear*g2shear + m1_dmu_dg1.even[m.M2] * g1shear * dmu + m1_dmu_dg2.even[m.M2]*g2shear*dmu + m1_dmu_dmu.even[m.M2]*dmu**2))
    print("MC: %s = %s" %(shiftedmoments4.even[m.MC] -2*shiftedmoments1.even[m.MC] + shiftedmoments4b.even[m.MC], m1_dg1_dg1.even[m.MC] * g1shear**2 + m1_dg2_dg2.even[m.MC] * g2shear**2 + 2.0*m1_dg1_dg2.even[m.MC] * g1shear*g2shear + m1_dmu_dg1.even[m.MC] * g1shear * dmu + m1_dmu_dg2.even[m.MC]*g2shear*dmu + m1_dmu_dmu.even[m.MC]*dmu**2))
    print("Looks Good!")
if (testrotation == True):
    import galsim
    phideg = 30.0
    gal1r = gal1.rotate(phideg*galsim.degrees)
    
    image_arr1r = galsimgen.return_array(gal1r,pixel_scale,stamp_size,noise_level = galaxy.noise_var, use_gaussian_noise = True,convolve_with_psf = True, psf = psf)
    kimage_arr1r, kx1r, ky1r, d2k1r, conjugate1r, kvar1r = momentcalc.simpleImage(image_arr1r,24,24,psf_arr,pixel_scale=pixel_scale,pixel_noise=np.sqrt(noise_var))

    moments1r = momentcalc.MomentCalculator(kimage_arr1r,kx1r,ky1r,d2k1r,conjugate1r,weight,kvar=kvar1r,id=0)
    if recenter:
        dx1r = moments1r.recenter()
    shiftedmoments1r=moments1r.get_moment(0,0)
    template1r=moments1r.get_template(0,0)
    m1r = template1r.get_moment()
    m1rot = m1.rotate(phideg*np.pi/180.)
    template1rot=template1.rotate(phideg*np.pi/180.)
    print("    ")
    print("Test Moment Rotation with phi = %s deg" %(phideg))
    print("original moment | rotated moment (calc) = rotated moment (galsim)")
    print("MF: %s | %s = %s" %(m1.even[0],m1rot.even[0],m1r.even[0]))
    print("MR: %s | %s = %s" %(m1.even[1],m1rot.even[1],m1r.even[1]))
    print("M1: %s | %s = %s" %(m1.even[2],m1rot.even[2],m1r.even[2]))
    print("M2: %s | %s = %s" %(m1.even[3],m1rot.even[3],m1r.even[3]))
    print("MC: %s | %s = %s" %(m1.even[4],m1rot.even[4],m1r.even[4]))
    print("MX: %s | %s = %s" %(m1.odd[0],m1rot.odd[0],m1r.odd[0]))
    print("MY: %s | %s = %s" %(m1.odd[1],m1rot.odd[1],m1r.odd[1]))

    print("  ")
    print("Test Template Rotation with phi = %s deg" %(phideg))
    for ii in xrange(template1.ND):

        if (ii == 0):
            print("Moments")
        if (ii == 1):
            print("1st deriv wrt mag DU")
        if (ii == 2):
            print("1st deriv wrt complex shear DV")
        if (ii == 3):
            print("1st deriv wrt complex shear conjugate DVb")
        if (ii == 4):
            print("2nd deriv wrt magnification DUDU")
        if (ii == 5):
            print("2nd deriv wrt magnification & complex shear DUDV")
        if (ii == 6):
            print("2nd deriv wrt magnification & complex shear conjuate DUDVb")
        if (ii == 7):
            print("2nd deriv wrt complex shear conjuate DVDV")
        if (ii == 8):
            print("2nd deriv wrt complex shear conjuate DVbDVb")
        if (ii == 9):
            print("2nd deriv wrt complex shear & conjuate DVDVb")

        print("original moment | rotated moment (calc) = rotated moment (galsim)")

        print("MF: %s | %s = %s" %(template1.even[0][ii].real,template1rot.even[0][ii].real,template1r.even[0][ii].real))
        print("MR: %s | %s = %s" %(template1.even[1][ii].real,template1rot.even[1][ii].real,template1r.even[1][ii].real))
        print("M1: %s | %s = %s" %(template1.even[3][ii].real,template1rot.even[3][ii].real,template1r.even[3][ii].real))
        print("M2: %s | %s = %s" %(template1.even[3][ii].imag,template1rot.even[3][ii].imag,template1r.even[3][ii].imag))
        print("MC: %s | %s = %s" %(template1.even[2][ii].real,template1rot.even[2][ii].real,template1r.even[2][ii].real))
        print("MX: %s | %s = %s" %(template1.odd[0][ii].real,template1rot.odd[0][ii].real,template1r.odd[0][ii].real))
        print("MY: %s | %s = %s" %(template1.odd[0][ii].imag,template1rot.odd[0][ii].imag,template1r.odd[0][ii].imag))
        pdb.set_trace()
