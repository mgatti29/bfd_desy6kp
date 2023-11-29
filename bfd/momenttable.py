# classes for collecting target / template moments and saving to FITS

import sys
import numpy as np
from astropy import version 
import astropy.io.fits as fits
from . import momentcalc
from .keywords import *


class TargetTable(object):
    '''Class to save a list of target galaxies to FITS binary table
    '''
    def __init__(self, n, sigma, allCov = None, stampMode=False):
        '''
        `n, sigma:`  are parameters of the weight function
        `allCov:`       is a MomentCovariance or a matrix of the even moments' covariance.
                     which if given is then assumed the same for all targets.
        `stampMode:` is True if outputs are result of a known number of galaxy insertions.  If False,
                     the we're in "Poisson" mode, in which case we will need to register non-detection
                     objects into the catalog in order to calculate selection-bias terms.
        '''

        # Create lists for the stuff we're going to save
        # ??? do this with arrays ???
        self.moment = []
        self.id = []
        self.xy = []
        self.num_exp = []
        
        
        
        #Added by Marco
        self.psf_moment = []
        self.psf_hsm_moment = []
        self.psf_moment_obs = []
        self.cov_psf_obs = []
        self.cov_psf_model = []
        self.cov_psf_shotnoise = []   
        self.orig_row = []
        self.orig_col = []
        self.DESDM_coadd_y = []
        self.DESDM_coadd_x = []
        self.ccd_name = []
        self.bkg = []
        self.pixel_used_bkg = []
        self.cov_Mf_per_band = []
        self.meb = []     
        self.mfrac_per_band = []
        self.good_exposures = []
        self.bad_exposures = []
        self.psf_hsm_moments_obs = []
        self.stampMode = stampMode

        # Make an array for areas of non-detections if we need it
        if self.stampMode:
            # In stamp mode, we only need area if someone shows up with area that isn't 1 or 0
            self.area = None
        else:
            self.area = []

        if allCov is None:
            # Need a cov for each target
            self.cov = []
        elif type(allCov) is MomentCovariance:
            self.allCov= allCov.even.copy()
        else:
            self.allCov = np.array(allCov)

        # Potential column showing if galaxy is selected
        self.select = None
        
        # Store covariance and other info into a header
        prihdr = fits.Header()
        prihdr[hdrkeys['weightN']] = n
        prihdr[hdrkeys['weightSigma']]=sigma
        prihdr['NCOLOR'] = 0   # Not putting colors in yet

        if self.stampMode:
            prihdr['STAMPS'] = 1
        else:
            prihdr['STAMPS'] = 0

        # Store covariance matrix in primary extension if it's universal
        if allCov is not None:
            self.prihdu = fits.PrimaryHDU(cov[0],header=prihdr)
        else:
            self.prihdu = fits.PrimaryHDU(header=prihdr)

        return

    def add(self, mom, xy=(0.,0.), id=0, covgal = None):
        '''Add a measured target to the table, with its MomentCovariance if they
        are being individuated'''

        # An error if covariance is given here there is a general one,
        # or if there are neither
        #if covgal is not None and self.cov is not None:
        #    raise ValueError('Giving covariance for a single galaxy when there is' +\
        #                     'a global covariance for the TargetTable')
        if covgal is None and self.cov is None:
            raise ValueError('Missing covariance for a target')

        # Add information for another target
        self.moment.append(mom.even)
        self.id.append(id)
        self.xy.append(np.array(xy))
        if self.area is not None:
            self.area.append(0.)

        if self.select is not None:
            # Don't know at this point if selected..
            self.select = 0  
        if covgal is not None:
            # Add packed covariance
            self.cov.append(covgal.pack())

        return

    def addNondetection(self, covND=None, area=0., xy=(0.,0.), id=0):
        '''Add a row indicating non-detection over a certain area.
        `covND` is the MomentCovariance at this location, if it's
               not universal for all targets.
        `area` is the area over which non-detection applies, which
               should be 0 or 1 when in stamp mode, or the actual
               area for Poisson mode.'''

        # An error if covariance is given here there is a general one,
        # or if there are neither
        if covND is not None and self.cov is not None:
            raise ValueError('Giving covariance for a single galaxy when there is' +\
                             'a global covariance for the TargetTable')
        if covND is None and self.cov is None:
            raise ValueError('Missing covariance for a target')

        # Add information to table
        if self.area is None and not (area==0. or area==1.):
            # We need to create the area column if this is
            # the first time we have a not-single-stamp area
            self.area = np.zeros(self.id.shape[0], dtype=np.float32)
        if self.area is not None:
            self.area.append(area)
        self.moment.append(Moment().even)  # Should be zeros
        self.id.append(id)
        self.xy.append(np.array(xy))
            
        if self.select is not None:
            # Don't know at this point if selected..
            self.select = 0  

        if covND is not None:
            # Add packed covariance
            self.cov.append(covND.pack())

        return

    def save(self, fitsname, overwrite=False):

        col=[]
        col.append(fits.Column(name="id",format="K",array=self.id))
        col.append(fits.Column(name="moments",format="5E",array=self.moment.astype(np.float32)))
        col.append(fits.Column(name="xy", format="2D", array=self.xy))
        if self.cov is not None:
            col.append(fits.Column(name=colnames['covariance'],format="15E",array=self.cov.astype(np.float32)))
        if self.area is not None:
            col.append(fits.Column(name=colnames['area'],format="E",array=self.cov.astype(np.float32)))
        if self.select is not None:
            col.append(fits.Column(name="select",format="I",array=self.cov.astype(np.int16)))
            
        cols=fits.ColDefs(col)
        tbhdu = fits.BinTableHDU.from_columns(cols)

        thdulist = fits.HDUList([self.prihdu,tbhdu])
        thdulist.writeto(fitsname,overwrite=overwrite)
        return

    @staticmethod
    def load(cls, fitsname):
        '''Create a TargetTable object from stored FITS array'''
        # ??? No provision yet for universal covariance ???
        ff = pf.open(fitsname)
        
        if ff[0].data.ndim==2:
            # There is a universal covariance
            cov = ff[0].data
        else:
            cov = None
        stampMode = ff[1].header['STAMPS']>0
        n = ff[1].header[hdrkeys['weightN']]
        sigma = ff[1].header[hdrkeys['weightSigma']]
        out = cls(n, sigma, allCov=cov, stampMode=stampMode)
        
        tab = ff[1].data
        out.id = np.array(tab['ID'])
        out.xy = np.array(tab['XY'])
        out.moments = np.array(tab['MOMENTS'])
        if 'covariance' in tab.names or colnames['covariance'] in tab.names:
            out.cov = np.array(tab[colnames['covariance']])
        else:
            out.cov = None
        if 'area' in tab.names or colnames['area'] in tab.names:
            out.area = np.array(tab[colnames['area']])
        else:
            out.area = None
        if 'select' in tab.names or 'SELECT' in tab.names:
            out.select = np.array(tab['SELECT'])
        else:
            out.select = None
        
        ff.close()
        return out
        
class TemplateTable(object):
    '''Class to save a list of template galaxies to FITS binary table
    '''
    def __init__(self, n, sigma, sn_min, sigma_xy, sigma_flux, sigma_step, sigma_max,**kwargs):
        # Save configuration info in a FITS header
        self.hdr = fits.Header()
        self.hdr[hdrkeys['weightN']] = n
        self.hdr[hdrkeys['weightSigma']]= sigma
        self.hdr[hdrkeys['fluxMin']]= sn_min * sigma_flux
        self.hdr[hdrkeys['sigmaXY']]= sigma_xy
        self.hdr[hdrkeys['sigmaFlux']]= sigma_flux
        self.hdr[hdrkeys['sigmaStep']]= sigma_step
        self.hdr[hdrkeys['sigmaMax']]= sigma_max

        self.templates = []
        return

    def add(self, tmpl):
        self.templates.append(tmpl)
        return

    def save(self, fitsname):
        savemoments=[]
        savemoments_dg1 = []
        savemoments_dg2 = []
        savemoments_dmu = []
        savemoments_dg1_dg1 = []
        savemoments_dg1_dg2 = []
        savemoments_dg2_dg2 = []
        savemoments_dmu_dg1 = []
        savemoments_dmu_dg2 = []
        savemoments_dmu_dmu = []
        id = []
        nda = []
        jSuppression = []

        for tmpl in self.templates:
            # obtain moments and derivs
            m0 = tmpl.get_moment()
            m1_dg1 = tmpl.get_dg1()
            m1_dg2 = tmpl.get_dg2()
            m1_dmu = tmpl.get_dmu()
            m2_dg1_dg1 = tmpl.get_dg1_dg1()
            m2_dg1_dg2 = tmpl.get_dg1_dg2()
            m2_dg2_dg2 = tmpl.get_dg2_dg2()
            m2_dmu_dg1 = tmpl.get_dmu_dg1()
            m2_dmu_dg2 = tmpl.get_dmu_dg2()
            m2_dmu_dmu = tmpl.get_dmu_dmu()
            # append to each list, merging even and odd moments
            savemoments.append(np.append(m0.even,m0.odd))
            savemoments_dg1.append(np.append(m1_dg1.even,m1_dg1.odd))
            savemoments_dg2.append(np.append(m1_dg2.even,m1_dg2.odd))
            savemoments_dmu.append(np.append(m1_dmu.even,m1_dmu.odd))
            savemoments_dg1_dg1.append(np.append(m2_dg1_dg1.even,m2_dg1_dg1.odd))
            savemoments_dg1_dg2.append(np.append(m2_dg1_dg2.even,m2_dg1_dg2.odd))
            savemoments_dg2_dg2.append(np.append(m2_dg2_dg2.even,m2_dg2_dg2.odd))
            savemoments_dmu_dg1.append(np.append(m2_dmu_dg1.even,m2_dmu_dg1.odd)) 
            savemoments_dmu_dg2.append(np.append(m2_dmu_dg2.even,m2_dmu_dg2.odd)) 
            savemoments_dmu_dmu.append(np.append(m2_dmu_dmu.even,m2_dmu_dmu.odd)) 
            nda.append(tmpl.nda)
            id.append(tmpl.id)
            jSuppression.append(tmpl.jSuppression)

        # Create the primary and table HDUs
        col1 = fits.Column(name="id",format="K",array=id)
        col2 = fits.Column(name="moments",format="7E",array=savemoments)
        col3 = fits.Column(name="moments_dg1",format="7E",array=savemoments_dg1)
        col4 = fits.Column(name="moments_dg2",format="7E",array=savemoments_dg2)
        col5 = fits.Column(name="moments_dmu",format="7E",array=savemoments_dmu)
        col6 = fits.Column(name="moments_dg1_dg1",format="7E",array=savemoments_dg1_dg1)
        col7 = fits.Column(name="moments_dg1_dg2",format="7E",array=savemoments_dg1_dg2)
        col8 = fits.Column(name="moments_dg2_dg2",format="7E",array=savemoments_dg2_dg2)
        col9 = fits.Column(name="moments_dmu_dg1",format="7E",array=savemoments_dmu_dg1)
        col10 = fits.Column(name="moments_dmu_dg2",format="7E",array=savemoments_dmu_dg2)
        col11= fits.Column(name="moments_dmu_dmu",format="7E",array=savemoments_dmu_dmu)
        col12= fits.Column(name="weight",format="E",array=nda)
        col13= fits.Column(name="jSuppress",format="E",array=jSuppression)
        cols = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13])
        tbhdu = fits.BinTableHDU.from_columns(cols, header=self.hdr)
        prihdu = fits.PrimaryHDU()
        thdulist = fits.HDUList([prihdu,tbhdu])
        thdulist.writeto(fitsname,overwrite=True)

        return
