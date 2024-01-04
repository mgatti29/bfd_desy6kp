# Extensions of momentcalc functions to galaxies with multiple stamps
import numpy as np
from .momentcalc import MomentCalculator, KData, Moment, Template, simpleImage
import pdb

class MultiMomentCalculator(MomentCalculator):
    ''' Class to calculate moments (or derivatives or covariances) from 
    multiple given flattened arrays, each PSF-corrected Fourier-domain samples
    of the surface brightness pattern for a given galaxy.

    Class implements all the methods of MomentCalculator, either because
    of inheritance or through an override.
    '''
    
    def __init__(self, kdata, weight, id=0, nda=1., img_weights=None, bandinfo=None):
        '''
        kdata = a single KData instance, or an array of them from different images
        weight = a weight-function generating class instance
        id   = integer identifying parent image
        nda   = sky density of objects like this (only relevant for templates)
        img_weights = array of weights to assign to each image's data.  If None,
                      weights will be assigned using covariance matrices.
        bandinfo = a dictionary containing the bands provided to the code, 
                   their weights, and an index for indexing the output moments 
                   for each band individually
        '''
        # general attributes to be saved for Multi class
        self.nda = nda
        self.weight = weight
        self.id = int(id)
        self.error_status = None
        try:
            self.kdata = list(kdata)
        except Exception:
            # Exception if kdata was not iterable, so it's just one
            self.kdata = [kdata]
        self.nimage = np.shape(kdata)[0]

        # get imaging band info if provided
        band_list=[]
        for kd in range(self.nimage):
            band_list.append(self.kdata[kd].band)
        # check all values are None or no values are None
        allnone = all(b is None for b in band_list)
        nonone = all(b is not None for b in band_list)
        if allnone:
            # all band information is None so only one band provided
            self.nbands=1
            self.bands=None
            self.bandinfo=None
            self.band_list=np.array(band_list)
        else:
            # check if all band info provided
            if nonone:
                # all bands provided, get the band information#
                self.band_list=np.array(band_list)
                self.bands=set(band_list)
                self.nbands=len(self.bands)
                if bandinfo is None:
                    raise Exception("Must provide Band info if providing multiple bands")
                else:
                    # check same bands in bandinfo as in list - required to be same
                    self.bandinfo=bandinfo
                    if self.bands != set(bandinfo['bands']):
                        self.error_status="NotAllBandsProvided"
            else:
                # some band info provided some not, raise Exception
                raise Exception("If giving more than one band, cannot have None in band list")

        # set up a series of MomentCalculators
        self.mc = [MomentCalculator(kd,self.weight,self.id,self.nda) for kd in self.kdata]
        
        if img_weights is None:
            self.img_weights = self._get_img_weights()
        else:
            if len(img_weights) != len(kdata):
                raise Exception('MultiMomentCalculator weight length does not match # of images')

        self.img_weights = np.array(self.img_weights, dtype=float)


        return

    def _get_img_weights(self):
        '''returns array of normalized weights for individual images based on covariance
        if multiple bands provided, gives normalized weights for each individual band
        '''
        m = Moment()
        no_weight = (len(self.mc) == self.nbands)
        if no_weight:
            wt = np.array([1. for mc in self.mc])
        else:
            wt = np.array([1./mc.get_covariance()[0][m.M0][m.M0] for mc in self.mc])
        # normalize relative to each band
        if self.nbands > 1:
            for b in self.bandinfo['bands']:
                sel=np.where(self.band_list == b)
                wt[sel]=wt[sel]/np.sum(wt[sel])
        else:
            wt /= np.sum(wt)   # Normalize to unit sum
            
        return wt


    def get_moment(self, dx, dy, returnbands=False):
        '''Return weighted moment vector with coordinate origin at dx, dy
         '''
        m = Moment()
        mband=[Moment() for _ in range(self.nbands)]
        if self.bandinfo is not None:
            # loop over images to get each band's moments
            for ii,mc in enumerate(self.mc):
                mi = mc.get_moment(dx,dy)
                b = self.bandinfo['bands'].index(self.band_list[ii])
                mband[b].even += mi.even * self.img_weights[ii]
                mband[b].odd  += mi.odd * self.img_weights[ii]
            # loop over bands to get total moment
            for kk in range(self.nbands):
                m.even += mband[kk].even * self.bandinfo['weights'][kk]
                m.odd += mband[kk].odd * self.bandinfo['weights'][kk]

        else:
            for ii,mc in enumerate(self.mc):
                mi = mc.get_moment(dx,dy)
                m.even += mi.even * self.img_weights[ii]
                m.odd  += mi.odd * self.img_weights[ii]

        if returnbands:
            return m, mband
        else:
            return m

    def _set_shifted(self, dx, dy):
        '''Alter all k values with phase shifts induced
        by moving origin by (dx,dy)
        (routine is used by recenter() method of base class)
        '''
        for mc in self.mc:
            mc._set_shifted(dx,dy)
        return
    
    def get_covariance(self,returnbands=False):
        '''Return covariance matrix for ensemble of images for even and odd moment sets
        must normalize weights first
        '''
        m = Moment()
        covm_even = np.zeros((m.NE, m.NE), dtype=float)
        covm_odd  = np.zeros((m.NO, m.NO), dtype=float)
        covm_even_all = np.zeros((m.NE,m.NE,self.nbands),dtype=float)
        covm_odd_all  = np.zeros((m.NO,m.NO,self.nbands),dtype=float)

        if self.bandinfo is not None:
            # loop over images
            for ii,mc in enumerate(self.mc):
                covi=mc.get_covariance()
#                print ii, np.diagonal(covi[0])
                b=self.bandinfo['bands'].index(self.band_list[ii])
                covm_even_all[:,:,b] += covi[0] * self.img_weights[ii]**2
                covm_odd_all[:,:,b] += covi[1] * self.img_weights[ii]**2
 
            for kk in range(self.nbands):
 
                b=self.bandinfo['index'][kk]
                covm_even += covm_even_all[:,:,b] *self.bandinfo['weights'][kk]**2
                covm_odd  += covm_odd_all[:,:,b] *self.bandinfo['weights'][kk]**2
 
        else:
            for ii,mc in enumerate(self.mc):
                covi = mc.get_covariance()
                covm_even += covi[0] * (self.img_weights[ii]**2)
                covm_odd  += covi[1] * (self.img_weights[ii]**2)

        if returnbands:
            return covm_even,covm_odd,covm_even_all,covm_odd_all
        else:
            return covm_even, covm_odd

    def get_template(self,dx,dy,returnbands=False):
        '''Return Template object for ensemble of images (moments and their derivatives) 
        with origin shifted by dx,dy.
        '''
        t = Template() 
        tall=Template()
        tband=[Template() for _ in range(self.nbands)]

        even_out = np.zeros( (t.NE,t.ND), dtype=np.complex64)
        odd_out  = np.zeros( (t.NO,t.ND), dtype=np.complex64)
        
        even_out_band = np.zeros( (t.NE,t.ND,self.nbands),dtype=np.complex64)
        odd_out_band  = np.zeros( (t.NO,t.ND,self.nbands),dtype=np.complex64)

        if self.bandinfo is not None:
            # start loop
            for ii,mc in enumerate(self.mc):
                t = mc.get_template(dx,dy)
                b=self.bandinfo['bands'].index(self.band_list[ii])
                even_out_band[:,:,b] += t.even * self.img_weights[ii]
                odd_out_band[:,:,b]  += t.odd  * self.img_weights[ii]
            for kk in range(self.nbands):
                b=self.bandinfo['index'][kk]
                even_out += even_out_band[:,:,b]*self.bandinfo['weights'][kk]
                odd_out  += odd_out_band[:,:,b]*self.bandinfo['weights'][kk]
                tband[kk]=Template(self.id,even_out_band[:,:,b],odd_out_band[:,:,b],nda=self.nda)
            tall=Template(self.id,even_out,odd_out,nda=self.nda)
        else:
            for ii, mc in enumerate(self.mc):
                t=mc.get_template(dx,dy)
                even_out += t.even * self.img_weights[ii]
                odd_out  += t.odd * self.img_weights[ii]
            tall=Template(self.id,even_out,odd_out,nda=self.nda)
        # Done!
        if returnbands:
            return tall,tband
        else:
            return tall




#def multiImage(imagelist, origin, psflist, wcslist, pad_factor=1., bandlist=None, pixel_noiselist=None):
def multiImage(imagelist, origin, psflist, wcslist, pad_factor=1.,bandlist=None, pixel_noiselist=None,psf_recenter_sigma=0., eta = 1, shot_noise = False):
    '''Create PSF-corrected k-space images and variance arrays for multiple
    image postage stamps of an object.  
           2nd axis taken as x.
    numims number of images in given list
    imagelist:  list of numpy 2d arrays, in units of FLUX PER PIXEL, x coord is 2nd index.
    origin:     Location of the origin of the galaxy (2 element array) in the (common) world
                coordinates of all the images.  Should be near the center of each image.
    psflist:    matching-size postage stamps for the PSF corresponding to each image.  
                We will fix normalization. Each assumed to have origin at [N/2,N/2] pixel.
    wcslist:    WCS objects to use for each image.
    pad_factor: is (integer) zero-padding factor for each array before FFT's
    pixel_noiselist: array of RMS noise of the image pixel values.  Defaults to None.
    bandlist:   band specification for each input image.  Defaults to None.
    psf_recenter_sigma:  Size (in pixels) of Gaussian window used to recenter PSFs.
                <=0 will assume PSF is centered at (N//2, N//2) of image.
    Returns list of KData instances for each input image
    '''
    kdatalist=[]
    psf_shifts = []
    shot_noise_cov = []
    for ii,img in enumerate(imagelist):
        if pixel_noiselist is None:
            pixel_noise=None
        else:
            pixel_noise=pixel_noiselist[ii]
        
        if bandlist is None:
            band = None
        else:
            band = bandlist[ii]
        
        kdata_,psf_shift,shot_noise_cov_i  = simpleImage(img, origin,
                                     psflist[ii],
                                     wcs = wcslist[ii],
                                     pad_factor = pad_factor,
                                     pixel_noise=pixel_noise,
                                     band=band,psf_recenter_sigma=psf_recenter_sigma, eta = eta, shot_noise = shot_noise)
        
        kdatalist.append(kdata_)
        psf_shifts.append(psf_shift)
        shot_noise_cov.append(shot_noise_cov_i)
        

    return kdatalist,psf_shifts
