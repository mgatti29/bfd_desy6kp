'''
class to generate ellipticity distribution 
class to generate galaxy objects in GalSim (uses Christina's set of params)
function to produce psfs in Galsim
function to draw real space image with Galsim
function to draw k space image with Galsim
'''
import sys
import os
import math
import logging
import galsim
import numpy as np
import pdb


class BobsEDist:
    """
    Sets up an ellipticity distribution: exp^(-e^2/2sig^2)*(1-e^2)^2
    Pass a UniformDeviate as ud, else one will be created and seeded at random.
    """
    def __init__(self, sigma, ud=None):
        self.ellip_sigma = sigma
        if ud is None:
            self.ud = galsim.UniformDeviate()
        else:
            self.ud = ud
        self.gd = galsim.GaussianDeviate(self.ud, sigma=self.ellip_sigma)

    def sample(self):
        if self.ellip_sigma==0.:
            e1,e2 = 0.,0.
        else:
            while True:
                while True:
                    e1 = self.gd()
                    e2 = self.gd()
                    esq = e1*e1 + e2*e2
                    if esq < 1.:
                        break
                if self.ud() < (1-esq)*(1-esq) :
                    break
        return e1,e2


class GalaxyGenerator:
    '''initializing info, for generating fake galaxies with galsim'''
    def __init__(self, e_sigma, flux_range, hlr_range, noise_var,
                 pixel_scale=1.0, g=(0.,0.), seed=0, **kwargs):
        self.e_sigma         = e_sigma
        self.flux_range      = flux_range
        self.hlr_range       = hlr_range
        self.noise_var       = noise_var
        self.g1              = g[0]
        self.g2              = g[1]
        self.es              = [0.0,0.0]
        self.ud              = galsim.UniformDeviate(seed)
        self.ellipdist       = BobsEDist(self.e_sigma,self.ud)
        self.pixel_scale     = pixel_scale
        self.fixsersic       = kwargs['fixsersic']

    def sample(self):

        # set up bulge/disk/galaxy ellipticity params

        bulge_frac = self.ud()
        flux       = self.ud()*(self.flux_range[1]-self.flux_range[0]) + self.flux_range[0]
        hlr        = self.ud()*(self.hlr_range[1]-self.hlr_range[0]) + self.hlr_range[0]
        bulge      = galsim.DeVaucouleurs(half_light_radius = hlr)
        disk       = galsim.Exponential(half_light_radius = hlr)
        gauss      = galsim.Gaussian(half_light_radius = hlr)
        self.es    = self.ellipdist.sample()
        bulge      = bulge.shear(e1=self.es[0],e2=self.es[1])
        disk       = disk.shear(e1=self.es[0],e2=self.es[1])
        gauss      = gauss.shear(e1=self.es[0],e2=self.es[1])

        # set up galaxy offseting
        dxd        = (self.pixel_scale/2.) * (self.ud()*2.0-1.0)
        dyd        = (self.pixel_scale/2.) * (self.ud()*2.0-1.0)
        disk       = disk.shift(dxd,dyd)
        gauss      = gauss.shift(dxd,dyd)
        theta      = self.ud()*(2*np.pi)
        rr         = self.ud()*hlr #/2.
        dxb        = rr * np.cos(theta) + dxd
        dyb        = rr * np.sin(theta) + dyd
        bulge      = bulge.shift(dxb,dyb)

        # make combined galaxy
        if (self.fixsersic == 0):
            gal        = bulge_frac*bulge + (1.0-bulge_frac)*disk

        if (self.fixsersic == 1):
            gal        = disk

        if (self.fixsersic == 2):
            gal        = bulge

        if (self.fixsersic == 3):
            gal        = gauss

        # set flux value of galaxy
        gal = gal.withFlux(flux)

        # if applying a shear, apply shear
        if ((self.g1 != 0.0) | (self.g2 != 0.0)):
            gal = gal.shear(g1=self.g1,g2=self.g2)

        return gal

    def nominal(self, flux=1.):
        # Return a galaxy with nominal properties
        if self.fixsersic == 0:
            bulge_frac = 0.5
        if self.fixsersic == 1:
            bulge_frac = 0.0
        if self.fixsersic == 2:
            bulge_frac = 1.0

        hlr        = np.mean(self.hlr_range[1])
        bulge      = galsim.DeVaucouleurs(half_light_radius = hlr)
        disk       = galsim.Exponential(half_light_radius = hlr)
        gauss      = galsim.Gaussian(half_light_radius = hlr)

        # make combined galaxy with flux value
        if self.fixsersic >=0 and self.fixsersic <=2:
            gal        = bulge_frac*bulge + (1.0-bulge_frac)*disk
        if self.fixsersic == 3:
            gal        = gauss
            
        gal = gal.withFlux(flux)

        return gal

def define_psf(type, args, e=[0.0,0.0], **kwargs):
    '''
    code to define the psf, for now limiting to
    Airy, Gaussian, or Moffat
    Airy takes two arguments (wavelength in nm, tel diameter in m)
    Gaussian takes one argument (setting to fwhm in pixels)
    Moffat takes two arguments (half_light_radius in pixels and beta)
    '''
    gdtype = (type == "Airy") | (type == "Gaussian") | (type == "Moffat")
    if (gdtype == False):
        raise Exception("Invalid psftype provided, check config file")

    if (type == "Airy"):
        if len(args)<2:
            raise Exception("Insufficient arguments for Airy disk PSF")
        if ((args[0] > 0.0) & (args[1] > 0.0)):
            psf = galsim.Airy(lam = args[0], diam = args[1], scale_unit=galsim.arcsec)
        else:
            raise Exception("Arguments %s and %s out of range for Airy disk PSF" %(args[0], args[1]))

    if (type == "Gaussian"):
        if len(args)<1:
            raise Exception("Insufficient arguments for Gaussian PSF")
        if (args[0] > 0.0):
            psf = galsim.Gaussian(fwhm = args[0])
        else:
            raise Exception("Argument %s out of range for Gaussian PSF" %(args[0]))
            
    if (type == "Moffat"):
        if len(args)<2:
            raise Exception("Insufficient arguments for Moffat PSF")
        if ((args[0] > 0.0) & (args[1] > 0.0)):
            psf = galsim.Moffat(half_light_radius = args[0], beta = args[1])
        else:
            raise Exception("Arguments %s and %s out of range for Moffat PSF" %(args[0], args[1]))

    logging.info("%s psf with FWHM = %s" %(type,psf.fwhm))    

    if e[0]!=0. or e[1]!=0.:
        psf = psf.shear(e1 = e[0],e2 = e[1])
        logging.info("applied ellipiticity of e1=%s, e2=%s to PSF" %(e[0],e[1]))

    return psf

def return_karray(obj, pixel_scale, image_size,
                  dk=1.0, noise_var = 0, use_gaussian_noise = True, noise_seed = 0,
                  return_obj = False, convolve_with_psf = False, psf = 0, **kwargs):
    '''
    function to return the Fourier space image as a 
    numpy array using Galsim's drawKImage
    imsz provides the size of the image
    pixelscale gives realspace pixel scale - think this needs 
    to be inverted for the Fourier image
    '''
    if (convolve_with_psf == True):
        final = galsim.Convolve([obj,psf])
    else:
        final = obj
        
    finalim = final.drawKImage(nx=image_size,ny=image_size,scale=dk)

    finalimall = np.zeros((image_size,image_size),dtype="complex128")
    finalimall.real=finalim[0].array
    finalimall.imag=finalim[1].array

    # ??? Add appropriate noise ???
    if (return_obj == True):
        return finalimall,finalim
    else:
        return finalimall


def return_array(obj, pixel_scale, image_size,
                 noise_var = 0.0, use_gaussian_noise = True, noise_seed = 0,
                 return_obj=False, convolve_with_pixel=True, convolve_with_psf = False,
                 psf = 0, **kwargs):
    '''
    function to return the real space image as a 
    numpy array with Galsim's drawImage
    obj is the Galsim object
    pixel_scale is the scale of the pixels in arcsec
    image_size is the desired size of the image in pixels
    noise_var is the variance of the desired noise to apply
      if 0 no noise applied
      if > 0, apply noise (not literally adding a background, but noise due to such a 
      background)
    use_gaussian_noise says whether to use Gaussian with sigma = sqrt(noiselevel)
      or Poisson noise with level = noiselevel
    noise_seed = initates rng for background 
    return_obj is an option to return the Galsim object in addition to the array
    setting drawImage use_true_center=False, for odd arrays, this will still 
      place galaxy in central pixel but for even arrays will at upper right
      pixel of center
    convolve_with_pixel = True convolves with pixel profile
    convolve_with_psf = True will convolve with the psf (must be given as Galsim object)
    '''
    big_fft_params=galsim.GSParams(maximum_fft_size=24576)
    if (convolve_with_psf == True):
        if (psf == 0):
            raise Exception("Did not provide a valid psf")

        logging.debug("convolving galaxy with psf")
        final = galsim.Convolve([obj,psf],gsparams=big_fft_params)

    else:
        logging.debug("no convolution with psf")
        final = obj


    if (convolve_with_pixel == True):
        logging.debug("convolving with pixel profile")
        finalim = final.drawImage(nx=image_size,ny=image_size,scale=pixel_scale,use_true_center = False, method='auto')
    else:
        logging.debug("not convolving with pixel profile")
        finalim = final.drawImage(nx=image_size,ny=image_size,scale=pixel_scale,use_true_center = False, method='no_pixel')


    if (noise_var > 0):
        
        if (use_gaussian_noise == False):
            logging.debug("Poisson noise of Var ~ %s added" %(noise_var))
            noise=galsim.PoissonNoise(galsim.BaseDeviate(noise_seed),sky_level=noise_var)

        else:
            logging.debug("Gaussian noise of Var ~ %s added" %(noise_var))
            noise=galsim.GaussianNoise(galsim.BaseDeviate(noise_seed),sigma = np.sqrt(noise_var))

        finalim.addNoise(noise)

    if (return_obj == True):
        return finalim.array, finalim
    else:
        return finalim.array

    ''' rotate to be in normal convention
    if (retobj):
        return np.rot90(objim.array), objim
    else:
        return np.rot90(objim.array)
    '''
