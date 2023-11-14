'''
class to set compute moments
 sets psf, weight function, computes covariance matrix, and moments, and finds
 x and y offsets
functions to compute XY moment and its derivative
'''
import sys
import os
import math
import logging
import numpy as np
from scipy.optimize import fsolve
import pdb
import copy

version_split_string = (np.version.version).split('.')
numpy_version = np.float(version_split_string[1])

# A weight function class.  The attributes for derivatives are
# calculated when needed.
class KSigmaWeight(object):
    '''Calculate the k-sigma weight function.  The interface should
    be the same for other weight functions, aside from n and weight_sigma.
    Arguments:
    kx, ky: arrays holding kx and ky values (in sky units) at sampled points.
            Must have same shape.  Calculated values will have same shape too.
    n, sigma: Weight is defined as (1-k^2 sigma^2/(2*n))**n
    Attributes:
    mask:   boolean for locations of nonzero weights
    w:      The weight function
    dw:     First deriv w.r.t k^2
    d2w:    Second deriv
    '''
    def __init__(self, n=4, sigma=1.5, **kwargs):
        self.n = n
        self.sigsq2 = 0.5*sigma*sigma
        self._dw = None   #  Calculate on demand
        self._d2w = None   #  Calculate on demand
        return
    def set_k(self, kx, ky):
        if kx.shape != ky.shape:
            raise Exception('kx and ky shapes do not match')
        # Could check here for repetition of previous args to prevent repetition
        self.term = 1. - self.sigsq2*(kx*kx+ky*ky) / self.n
        self.mask = self.term>0.
        self.w = np.where(self.mask, self.term**self.n, 0.)
        self._dw = None
        self._d2w = None
    @property
    def dw(self):
        # First derivative of weight with respect to k^2
        if self._dw is None:
            self._dw = (-self.sigsq2)*np.where(self.mask, self.term**(self.n-1), 0.)
        return self._dw
    @property
    def d2w(self):
        # Second derivative of weight with respect to k^2
        if self._d2w is None:
            self._d2w = (self.sigsq2*self.sigsq2*(self.n-1)/(self.n)) \
              * np.where(self.mask, self.term**(self.n-2), 0.)
        return self._d2w
    def clone(self):
        # Return deep copy of self
        result = KSigmaWeight(self.n)
        result.sigsq2 = self.sigsq2
        if hasattr(self,'mask'):
            result.mask = self.mask
        if hasattr(self,'term'):
            result.term = self.term
        if hasattr(self,'w'):
            result.w = self.w
        if hasattr(self,'_dw'):
            result._dw = self._dw
        if hasattr(self,'_d2w'):
            result._d2w = self._d2w
        return result

class KBlackmanHarris:
    '''Calculate the Blackman-Harris weight function. The B-H function is
    extended to 2 dimensions by making it a function of radius.
    Arguments:
    kx, ky: arrays holding kx and ky values (in sky units) at sampled points.
            Must have same shape. Calculated values will have same shape too.
    sigma: sets the RMS size per dimension of the real-space version of wt func
    Attributes:
    kr: radius in k-space defined as (hypot(kx,ky))
    mask: boolean for locations of nonzero weights
    w: The weight function defined as 
                0.35875 
                + 0.48829*cos(pi*(kr/kmax))
                + 0.14128*cos(2*pi*(kr/kmax))
                + 0.01168*cos(3*pi*(kr/kmax))
            for kr<kmax, with kmax = 1.07635*np.pi/sigma, and 0 for k>=kmax
    dw: First deriv w.r.t kr^2
    d2w: Second deriv w.r.t kr^2
    '''
    coeffs = np.array([0.35875, 0.48829, 0.14128, 0.01168])
    m = np.arange(4) # Factor on each term
    def __init__(self, sigma=1.5, **kwargs):
        self.sigma = sigma
        self.kmax = 1.07635*np.pi/sigma
        self._dw = None # Calculate on demand
        self._d2w = None # Calculate on demand
        return

    def set_k(self, kx, ky):
        if kx.shape != ky.shape:
            raise Exception('kx and ky shapes do not match')
        # Could check here for repetition of previous args to prevent repetition
        self.kshape = kx.shape
        # Check that k space covers the full weight function
        if np.max(np.abs(kx)) < self.kmax or np.max(np.abs(ky)) < self.kmax:
            raise ValueError('k values do not reach edge of the weight function')
        
        # All internal calculations will flatten the array to make indexing simple
        kr = np.hypot(kx,ky).flatten()
        

        # Inverse of radial k where inf at origin is eliminated
        self._invkr = np.where( kr==0, 1., 1/kr).flatten()
        
        # This will be the argument of all the trig functions in B-H formula:
        self._u = kr * (np.pi / self.kmax)
        
         # Make mask that's true at pixels with non-zero weight
        self.mask_flat = (kr <= self.kmax)
        
        self._u_mask = self._u[self.mask_flat]
        
        self._cos_u_mask = np.cos(self._u_mask)
        self._cos_2u_mask = np.cos(2*self._u_mask)
        self._cos_3u_mask = np.cos(3*self._u_mask)
        
        self.w_mask = (self.coeffs[0]
                  + self.coeffs[1]*self._cos_u_mask
                  + self.coeffs[2]*self._cos_2u_mask
                  + self.coeffs[3]*self._cos_3u_mask)
        
        self.w = np.zeros(self.kshape).flatten()
        
        
        self.w[self.mask_flat] = self.w_mask
        self.w = self.w.reshape(self.kshape)
        self.mask = self.mask_flat.reshape(self.kshape)
        self._dw = None
        self._d2w = None
        
        
        
        
    @property
    def dw(self):
        # First derivative of weight with respect to v = (kr)^2
        # Noting that dw/dv = dw/du * du/dv = dw/du * (np.pi/ (2*kmax*kr))
        if self._dw is None:            
            self._dw_du_mask = (- self.coeffs[1] * np.sin(self._u_mask)
                                - 2 * self.coeffs[2] * np.sin(2*self._u_mask)
                                - 3 * self.coeffs[3] * np.sin(3*self._u_mask))

            self._dw_du = np.zeros(self.kshape).flatten()

            self._dw_du[self.mask_flat] = self._dw_du_mask

            self._dw_du = self._dw_du.reshape(self.kshape)

            self._dw = np.zeros(self.kshape).flatten()
            
            self._invkr_mask = self._invkr[self.mask_flat]

            self._dw[self.mask_flat] = (np.pi/(2*self.kmax)) * self._dw_du_mask * self._invkr_mask

            self._dw = self._dw.reshape(self.kshape)
        return self._dw
    
    @property
    def d2w(self):
        # Second derivative of weight with respect to kr^2.
        # Chain rule now gives d^2w/du^2 * (du/dv)**2 + dw/du * d^2u/dv^2
        # ...and d^2u/dv^2 = -3/4 / (kmax * k^3)
        if self._d2w is None:
        # Create 1st derivs if not done yet
            self.dw
            d2w_du2 = (-self.coeffs[1] * self._cos_u_mask
                       -4 * self.coeffs[2] * self._cos_2u_mask
                       -9 * self.coeffs[3] * self._cos_3u_mask)

            # 2nd deriv of u w.r.t. v=(kr)^2
            self._d2w_mask = (np.pi/(2*self.kmax))**2 * d2w_du2 * self._invkr_mask*self._invkr_mask \
                             + (-0.25 * np.pi / self.kmax) * self._dw_du_mask * self._invkr_mask*self._invkr_mask*self._invkr_mask


            self._d2w = np.zeros(self.kshape).flatten()
            self._d2w[self.mask_flat] = self._d2w_mask
            self._d2w = self._d2w.reshape(self.kshape)
        return self._d2w
    
    def clone(self):
        # Return deep copy of self
        result = KBlackmanHarris(self.sigma)
        if hasattr(self,'mask'):
            result.mask = self.mask
        if hasattr(self,'term'):
            result.term = self.term
        if hasattr(self,'w'):
            result.w = self.w
        if hasattr(self,'_dw'):
            result._dw = self._dw
        if hasattr(self,'_d2w'):
            result._d2w = self._d2w
        if hasattr(self,'kshape'):
            result.kshape = self.kshape
        if hasattr(self,'_u'):
            result._u = self._u
        if hasattr(self,'_dw_du'):
            result._dw_du = self._dw_du
        if hasattr(self,'mask'):
            result._mask = self.mask
        if hasattr(self,'_invkr'):
            result._invkr = self._invkr
        return result
    

class Moment(object):
    '''Hold the moments, split into the 5 evens and 2 odds.
    '''
    # Give read-only names to indices
    @property
    def M0(self):
        # Flux moment
        return 0
    @property
    def MR(self):
        # Radius moment
        return 1
    @property
    def M1(self):
        # E1 moment
        return 2
    @property
    def M2(self):
        # E2 moment
        return 3
    @property
    def MC(self):
        # k^4 (concentration) moment
        return 4
    @property
    def MX(self):
        # X moment - in the "odd" vector
        return 0
    @property
    def MY(self):
        # Y moment - in the "odd" vector
        return 1
    @property
    def NE(self):
        # Number of even moments
        return 5
    @property
    def NO(self):
        # Number of odd moments
        return 2
    def __init__(self, even=np.zeros(5,dtype=float),
                       odd=np.zeros(2,dtype=float)):
        self.even = np.array(even)
        self.odd = np.array(odd)
        if self.even.shape!=(self.NE,) or self.odd.shape!=(self.NO,):
            raise Exception('Wrong sized arrays for Moments')
        return
    def all(self):
        # Return single array of all moments
        return np.concatenate((self.even,self.odd))
    def rotate(self,phi):
        '''Return Moment instance for this object rotated by angle phi radians
        '''
        e = np.array(self.even)
        z = (self.even[self.M1] + 1j*self.even[self.M2]) * np.exp(2j*phi)
        e[self.M1] = z.real
        e[self.M2] = z.imag
        z = (self.odd[self.MX] + 1j*self.odd[self.MY]) * np.exp(1j*phi)
        o = np.array([z.real,z.imag])
        return Moment(e,o)
    def yflip(self):
        '''Return Moment instance for this object reflected about x axis
        y -> -y
        '''
        m = Moment(self.even, self.odd)
        m.even[m.M2] *= -1.
        m.odd[m.MY] *= -1.
        return m

class MomentCovariance(object):
    '''The covariance matrix for the components of a Moment object.
    It is assumed there is no covariance between the even and odd moments,
    so this object contains two distinct matrices for the even and the odd
    components.  Indexing is adopted from the Moment class.
    '''
    def __init__(self, even, odd=None):
        '''Expect two square numpy arrays for covariance matrices of
        the even and odd moments. Symmetry and pos-nonnegative-ness are not checked.
        If odd is omitted, it's constructed from the equivalent members of even.
        '''
        m = Moment()
        if not (len(even.shape)==2 \
                and even.shape[0]==m.NE \
                and even.shape[1]==m.NE):
            raise Exception('Even-parity covariance array has incorrect shape')
        self.even = np.array(even)
        if odd is None:
            # Make odd matrix from its even parts
            self.odd = np.zeros( (m.NO, m.NO), dtype=float)
            self.odd[m.MX, m.MX] = 0.5*(self.even[m.M0,m.MR]+self.even[m.M0,m.M1])
            self.odd[m.MY, m.MY] = 0.5*(self.even[m.M0,m.MR]-self.even[m.M0,m.M1])
            self.odd[m.MX, m.MY] = 0.5*self.even[m.M0,m.M2]
            self.odd[m.MY, m.MX] = 0.5*self.even[m.M0,m.M2]
        elif not (len(odd.shape)==2 \
                  and odd.shape[0]==m.NO \
                  and odd.shape[1]==m.NO):
            raise Exception('Odd-parity covariance array has incorrect shape')
        else:
            self.odd = np.array(odd)
        return
    def rotate(self,phi):
        '''Return covariance for this object rotated CCW by angle phi radians
        '''
        m = Moment()
        x = np.eye(m.NE, dtype=float)
        c = np.cos(2*phi)
        s = np.sin(2*phi)
        x[m.M1,m.M1] = c
        x[m.M1,m.M2] = -s
        x[m.M2,m.M1] = c
        x[m.M2,m.M2] = s
        e = np.dot( np.dot(x,self.even), x.T)

        x = np.eye(m.NO, dtype=float)
        c = np.cos(phi)
        s = np.sin(phi)
        x[m.MX,m.MX] = c
        x[m.MX,m.MY] = -s
        x[m.MY,m.MX] = c
        x[m.MY,m.MY] = s
        o = np.dot( np.dot(x,self.odd), x.T)
        return MomentCovariance(e,o)
    def yflip(self):
        ''' Return covariance of object reflected about the x axis,
        y -> -y
        '''
        m = Moment()
        e = np.array(self.even)
        e[:,m.M2] *= -1.
        e[m.M2,:] *= -1.
        o = np.array(self.odd)
        o[:,m.MY] *= -1.
        o[m.MY,:] *= -1.
        return MomentCovariance(e,o)
    def isotropize(self):
        '''Return a version of self with all non-monopole components nulled'''
        m = Moment()
        e = np.array(self.even)
        e[:,m.M1] = 0.
        e[m.M1,:] = 0.
        e[:,m.M2] = 0.
        e[m.M2,:] = 0.
        e[m.M1, m.M1] = e[m.M2,m.M2] = 0.5*e[m.MR,m.MR]
        o = np.zeros_like(self.odd)
        o[m.MX,m.MX] = o[m.MY,m.MY] = 0.5*(self.odd[m.MX,m.MX]+self.odd[m.MY,m.MY])
        return MomentCovariance(e,o)
        
    def pack(self):
        # Return a one-dimensional array of unique coefficients
        m = Moment()
        out = []
        for ii in range(m.NE):
            out.extend(self.even[ii,ii:])
        return np.array(out)
    
    @classmethod
    def unpack(cls, packed):
        m = Moment()
        # Return a full even & odd matrix given packed form
        e = cls.bulkUnpack(packed.reshape(1,-1)).reshape(m.NE,m.NE)
        # Odd moments will be made in contructor
        return cls(e)

    @staticmethod
    def bulkUnpack(packed):
        '''Convert an Nx15 packed 1d version of even matrix into Nx5x5 array'''
        m = Moment()
        out = np.zeros( (packed.shape[0],m.NE,m.NE), dtype=float)
        j=0
        for i in range(m.NE):
            nvals = m.NE - i
            out[:,i,i:] = packed[:,j:j+nvals]
            out[:,i:,i] = packed[:,j:j+nvals]
            j += nvals
        return out
        
class Template(object):
    '''A template galaxy, having moments
    plus derivatives of moments with respect to magnification and
    shear.  Will be kept in a form where every term has simple phase
    shift under rotation.
    '''
    # Define all the derivatives as read-only indices
    @property
    def D0(self):
        # No derivatives - the moments themselves
        return 0
    @property
    def DU(self):
        # 1st deriv wrt mag
        return 1
    @property
    def DV(self):
        # 1st deriv wrt complex shear
        return 2
    @property
    def DVb(self):
        # 1st deriv wrt complex shear conjugate
        return 3
    @property
    def DUDU(self):
        # 2nd deriv wrt magnification
        return 4
    @property
    def DUDV(self):
        # 2nd deriv wrt mag & complex shear
        return 5
    @property
    def DUDVb(self):
        # 2nd deriv wrt mag & complex shear conjugate
        return 6
    @property
    def DVDV(self):
        # 2nd deriv wrt complex shear
        return 7
    @property
    def DVbDVb(self):
        # 2nd deriv wrt complex shear conjugate
        return 8
    @property
    def DVDVb(self):
        # 2nd deriv wrt complex shear & conjugate
        return 9
    @property
    def ND(self):
        # Number of derivatives
        return 10

    # Define indices for the moments, in multipole form
    @property
    def M0(self):
        # Flux moment (m=0)
        return 0
    @property
    def MR(self):
        # Radius moment (m=0)
        return 1
    @property
    def MC(self):
        # Concentration moment (m=0)
        return 2
    @property
    def ME(self):
        # Ellipticity moment (m=2)
        return 3
    @property
    def NE(self):
        # Number of (complex) even moments
        return 4
    @property
    def MX(self):
        # X/Y Moments (m=1), in array of odds
        return 0
    @property
    def NO(self):
        # Number of (complex) odd moments
        return 1
    def __init__(self, id=0, even=None, odd=None, nda=1., jSuppression=1.):
        '''Template has an integer id unique to the parent object,
        then even/odd matrices giving moments and derivatives in
        pure multipole formats.
        nda = product of sky density of this template times area in origin-shift
              space that this template represents.
        jSuppression = ratio of this galaxy's xy Jacobian determinant to that
              obtained at xy which null the Mxy moments.  This factor shows up
              in the likelihood expression too.
        '''
        # Do dimension checking of inputs
        # Should be getting NE x ND, NO x ND arrays
        if even is not None and not (len(even.shape)==2 and \
                                     even.shape[0]==self.NE and \
                                     even.shape[1]==self.ND):
            raise Exception("wrong shape for even derivative array")
        if odd is not None and not (len(odd.shape)==2 and \
                                     odd.shape[0]==self.NO and \
                                     odd.shape[1]==self.ND):
            raise Exception("wrong shape for odd derivative array")
        self.id = int(id)
        self.even = even
        self.odd = odd
        self.nda = nda
        self.jSuppression = jSuppression
        return
    def rotate(self, phi):
        '''Return new Template instance for this object rotated by phi radians
        '''
        # Collect powers of phi to apply to each moment & deriv
        # First make vector of exp(j*n*phi)
        ejnphi = np.exp(np.arange(5)*1j*phi)
        ve = np.ones(self.NE, dtype=np.complex64)
        ve[self.ME] = ejnphi[2];
        vo = np.ones(self.NO, dtype=np.complex64)
        vo[self.MX] = ejnphi[1];
        vd = np.ones(self.ND, dtype=np.complex64)
        vd[self.DUDV] = np.conj(ejnphi[2]) # m = -2
        vd[self.DUDVb] = ejnphi[2]
        vd[self.DV] = np.conj(ejnphi[2]) # m = -2
        vd[self.DVb] = ejnphi[2]
        vd[self.DVDV] = np.conj(ejnphi[4]) # m = -4
        vd[self.DVbDVb] = ejnphi[4] # m = -4
        return Template(self.id,
                        ve[:,np.newaxis] * self.even * vd[np.newaxis,:],
                        vo[:,np.newaxis] * self.odd * vd[np.newaxis,:],
                        nda = self.nda,
                        jSuppression=self.jSuppression)
    def yflip(self):
        '''Return new Template instance for this object reflected about x axis,
        y -> -y
        '''
        e = np.array(self.even)
        o = np.array(self.odd)
        # Even moments flip imag part
        e.imag = -e.imag
        # Odd moments flip real part
        o.real = -o.real
        return Template(self.id,e,o,nda=self.nda)

    # Following are routines to get real-values moments and derivatives.
    # Each returns a Moment structure filled with specified derivs of
    # each moment.
    def _to_moment(self,e,o):
        '''Convert complex-valued even, odd moments in Template order
        into real-valued moment vector in Moment order
        '''
        m = Moment()
        m.even[m.M0] = e[self.M0].real
        m.even[m.MR] = e[self.MR].real
        m.even[m.MC] = e[self.MC].real
        m.even[m.M1] = e[self.ME].real
        m.even[m.M2] = e[self.ME].imag
        m.odd[m.MX] = o[self.MX].real
        m.odd[m.MY] = o[self.MX].imag
        return m

    def get_moment(self):
        '''Return moments for this template
        '''
        return self._to_moment(self.even[:,self.D0], self.odd[:,self.D0])
    def get_dmu(self):
        '''Return derivs of moments wrt magnification
        '''
        return self._to_moment(self.even[:,self.DU],
                               self.odd[:,self.DU])
    
    def get_dg1(self):
        '''Return derivs of moments wrt g1 shear
        '''
        return self._to_moment((self.even[:,self.DV]+self.even[:,self.DVb]),
                               (self.odd[:,self.DV]+self.odd[:,self.DVb]))
    def get_dg2(self):
        '''Return derivs of moments wrt g2 shear
        '''
        return self._to_moment(1j*(self.even[:,self.DV]-self.even[:,self.DVb]),
                               1j*(self.odd[:,self.DV]-self.odd[:,self.DVb]))
    def get_dmu_dmu(self):
        '''Return 2nd derivs of moments with respect to magnification
        '''
        return self._to_moment(self.even[:,self.DUDU], self.odd[:,self.DUDU])
    def get_dmu_dg1(self):
        '''Return 2nd derivs of moments wrt g1 shear & magnification
        '''
        return self._to_moment((self.even[:,self.DUDV]+self.even[:,self.DUDVb]),
                               (self.odd[:,self.DUDV]+self.odd[:,self.DUDVb]))
    def get_dmu_dg2(self):
        '''Return 2nd derivs of moments wrt g2 shear & magnification
        '''
        return self._to_moment(1j*(self.even[:,self.DUDV]-self.even[:,self.DUDVb]),
                               1j*(self.odd[:,self.DUDV]-self.odd[:,self.DUDVb]))
    def get_dg1_dg1(self):
        '''Return 2nd derivs of moments wrt g1 shear
        '''
        return self._to_moment(2*self.even[:,self.DVDVb]+self.even[:,self.DVDV] \
                                 +self.even[:,self.DVbDVb],
                               2*self.odd[:,self.DVDVb]+self.odd[:,self.DVDV] \
                                 +self.odd[:,self.DVbDVb])
    def get_dg2_dg2(self):
        '''Return 2nd derivs of moments wrt g2 shear
        '''
        return self._to_moment(2*self.even[:,self.DVDVb]-self.even[:,self.DVDV] \
                                 -self.even[:,self.DVbDVb],
                               2*self.odd[:,self.DVDVb]-self.odd[:,self.DVDV] \
                                 -self.odd[:,self.DVbDVb])
    def get_dg1_dg2(self):
        '''Return 2nd derivs of moments wrt g1,g2 shear
        '''
        return self._to_moment(1j*(self.even[:,self.DVDV]-self.even[:,self.DVbDVb]),
                               1j*(self.odd[:,self.DVDV]-self.odd[:,self.DVbDVb]))

        
        
class KData(object):
    '''Simple structure to hold k-space galaxy image information, already PSF-corrected.
    Arrays can be 1d or 2d but should all have the same shape.
        kval = PSF-corrected Fourier-domain surface-brightness samples (complex)
        kx, ky = values of k at samples
        d2k   = area in k-space per sample (scalar)
        conjugate = boolean array with True at each k for which we are integrating
                 over -k but it's not in the array.
        kvar = variance of each kval sample.  More precisely, Cov(kval, conj(kval)). Only
               needed if covariance matrix will be wanted.
    '''
    def __init__(self, kval, kx, ky, d2k, conjugate, kvar=None, band=None):
        '''Initializer makes internal names refer to same data as the arguments,
        so user is responsible that these arrays are not altered.
        '''
        self.kval = kval
        self.kx = kx
        self.ky = ky
        self.d2k = d2k
        self.conjugate = conjugate
        self.kvar = kvar
        self.band = band
        return
    
class MomentCalculator(object):
    ''' Class to calculate moments (or derivatives or covariances) given
    a flattened array of PSF-corrected Fourier-domain samples of the 
    surface brightness pattern.
    '''
    
    def __init__(self, kdata, weight, id=0, nda=1.):
        '''
        kdata = a KData instance
        weight = a weight-function generating class instance
        id   = integer identifying parent image
        nda   = sky density of objects like this (only relevant for templates)
        '''
        # These attributes do not change for each image
        self.nda = nda
        self.weight = weight.clone()
        self.weight.set_k(kdata.kx,kdata.ky)
        self.id = int(id)
        # Extract and save flattened, masked values, with area & conjugation factors
        # included
        self._kval = (kdata.kval * np.where(kdata.conjugate, 2., 1.))[self.weight.mask] * kdata.d2k
        self._kx = kdata.kx[self.weight.mask]
        self._ky = kdata.ky[self.weight.mask]
        if kdata.kvar is None:
            self._kvar = None
        else:
            # Save k-value variances, with conjugation and *2* factors of k-space area
            self._kvar = (kdata.kvar * np.where(kdata.conjugate, 2., 1.))[self.weight.mask] \
                             * (kdata.d2k*kdata.d2k)
        self._ksq = None         # |k^2|
        self._kz = None          # kx + j * ky

        self._wt_f_even = None   # Array of weight * moment kernel for all even moments
        self._wt_f_odd = None   # Array of weight * moment kernel for all odd moments

        return

    def _set_k(self):
        '''Set up more k stuff'''
        if self._ksq is None:
            self._ksq = self._kx*self._kx + self._ky*self._ky
        if self._kz is None:
            self._kz = self._kx + 1j * self._ky

    def _set_wt_f(self):
        ''' Calculate coefficients for moment integrals
        '''
        m = Moment()
        self._set_k()
        if self._wt_f_even is not None:
            return
        self._wt_f_even = np.zeros( (m.NE,)+self._kx.shape, dtype=float)
        self._wt_f_odd  = np.zeros( (m.NO,)+self._kx.shape, dtype=float)
        w = self.weight.w[self.weight.mask]
        self._wt_f_even[m.M0,:] = w
        self._wt_f_even[m.MR,:] = w * self._ksq
        kk = self._kz * self._kz
        self._wt_f_even[m.M1,:] = w * kk.real
        self._wt_f_even[m.M2,:] = w * kk.imag
        self._wt_f_even[m.MC,:] = w * self._ksq * self._ksq
        # For the odds, we have to remember there's a factor of 1j in front of these.
        self._wt_f_odd[m.MX,:] = w * self._kx
        self._wt_f_odd[m.MY,:] = w * self._ky
        return
        
    def _get_shifted(self, dx, dy):
        ''' Return a kval vector with phase shifts
        from moving origin by (dx,dy)
        '''
        phase = self._kx * dx + self._ky * dy
        return np.exp(1j*phase) * self._kval
    
    def _set_shifted(self, dx, dy):
        ''' Apply phase shifts to data in this object
        from moving origin by (dx,dy)
        '''
        self._kval = self._get_shifted(dx,dy)
    
    def get_moment(self, dx, dy):
        ''' Return moment vector with coordinate origin at dx, dy.
        '''
        
        self._set_wt_f()
        kval = self._get_shifted(dx,dy)
        even = np.sum(self._wt_f_even * kval.real, axis=1)
        odd  = -np.sum(self._wt_f_odd * kval.imag, axis=1)

        return Moment(even,odd)

    def get_covariance(self):
        '''Return covariance matrices for even and odd moment sets
        '''
        if self._kvar is None:
            raise Exception('get_covariance requires a kvar vector')
        self._set_wt_f()
        even = np.sum( self._wt_f_even[np.newaxis,:,:] * \
                       self._wt_f_even[:,np.newaxis,:] * \
                       self._kvar, axis=2) 
        odd  = np.sum( self._wt_f_odd[np.newaxis,:,:] * \
                       self._wt_f_odd[:,np.newaxis,:] * \
                       self._kvar, axis=2)
        return even, odd

    def get_template(self,dx,dy):
        '''Return Template object (moments and their derivatives) 
        with origin shifted by dx,dy.
        '''
        t = Template()
        # These tuples encode all the information about the formulae for
        # what goes into the derivatives.  Each tuple contains:
        # Index of moment
        # Index for derivative
        # Scalar prefactor
        # m value = p-q for k^p k.conj^q
        # N value = p+q

        # First those that are integrated with the weight function itself
        w0_terms_e = ( (t.M0, t.D0,  1.,  0, 0),
                       (t.MR, t.D0,  1.,  0, 2),
                       (t.MR, t.DU, -2.,  0, 2),
                       (t.MR, t.DV, -1., -2, 2),
                       (t.MR, t.DVb, -1., 2, 2),
                       (t.MR, t.DUDU, 6., 0, 2),
                       (t.MR, t.DUDV, 2., -2, 2),
                       (t.MR, t.DUDVb, 2., 2, 2),
                       (t.MR, t.DVDVb, 2., 0, 2),
                       (t.MC, t.D0,    1., 0, 4),
                       (t.MC, t.DU,   -4., 0, 4),
                       (t.MC, t.DV,   -2.,-2, 4),
                       (t.MC, t.DVb,  -2., 2, 4),
                       (t.MC, t.DUDU, 20., 0, 4),
                       (t.MC, t.DUDV,  8.,-2, 4),
                       (t.MC, t.DUDVb, 8., 2, 4),
                       (t.MC, t.DVDV,  2.,-4, 4),
                       (t.MC, t.DVDVb, 6., 0, 4),
                       (t.MC, t.DVbDVb,2., 4, 4),
                       (t.ME, t.D0,    1., 2, 2),
                       (t.ME, t.DU,   -2., 2, 2),
                       (t.ME, t.DV,   -2., 0, 2),
                       (t.ME, t.DUDU,  6., 2, 2),
                       (t.ME, t.DUDV,  4., 0, 2),
                       (t.ME, t.DVDV,  2.,-2, 2),
                       (t.ME, t.DVDVb, 1., 2, 2) )

        w0_terms_o = ( (t.MX, t.D0, 1.j, 1, 1),
                       (t.MX, t.DU, -1.j, 1, 1),
                       (t.MX, t.DV, -1.j, -1, 1),
                       (t.MX, t.DUDU, 2.j, 1, 1),
                       (t.MX, t.DUDV, 1.j,-1, 1),
                       (t.MX, t.DVDVb, 0.5j, 1, 1) )
            
        # Then terms using 1st deriv of W
        w1_terms_e = ( (t.M0, t.DU, -2.,  0, 2),
                       (t.M0, t.DV, -1., -2, 2),
                       (t.M0, t.DVb, -1., 2, 2),
                       (t.M0, t.DUDU, 6., 0, 2),
                       (t.M0, t.DUDV, 2.,-2, 2),
                       (t.M0, t.DUDVb,2., 2, 2),
                       (t.M0, t.DVDVb,2., 0, 2),
                       (t.MR, t.DU,  -2., 0, 4),
                       (t.MR, t.DV,  -1.,-2, 4),
                       (t.MR, t.DVb, -1., 2, 4),
                       (t.MR, t.DUDU, 14., 0, 4), 
                       (t.MR, t.DUDV,  6.,-2, 4), 
                       (t.MR, t.DUDVb, 6., 2, 4), 
                       (t.MR, t.DVDV , 2.,-4, 4), 
                       (t.MR, t.DVDVb, 4., 0, 4), 
                       (t.MR, t.DVbDVb,2., 4, 4), 
                       (t.MC, t.DU,  -2., 0, 6),
                       (t.MC, t.DV,  -1.,-2, 6),
                       (t.MC, t.DVb, -1., 2, 6),
                       (t.MC, t.DUDU, 22., 0, 6), 
                       (t.MC, t.DUDV, 10.,-2, 6), 
                       (t.MC, t.DUDVb,10., 2, 6), 
                       (t.MC, t.DVDV , 4.,-4, 6), 
                       (t.MC, t.DVDVb, 6., 0, 6), 
                       (t.MC, t.DVbDVb,4., 4, 6), 
                       (t.ME, t.DU,  -2., 2, 4),
                       (t.ME, t.DV,  -1., 0, 4),
                       (t.ME, t.DVb, -1., 4, 4),
                       (t.ME, t.DUDU, 14., 2, 4), 
                       (t.ME, t.DUDV, 10., 0, 4), 
                       (t.ME, t.DUDVb, 4., 4, 4), 
                       (t.ME, t.DVDV , 4.,-2, 4), 
                       (t.ME, t.DVDVb, 4., 2, 4) )

        w1_terms_o = ( (t.MX, t.DU, -2.j, 1, 3),
                       (t.MX, t.DV, -1.j,-1, 3),
                       (t.MX, t.DVb,-1.j, 3, 3),
                       (t.MX, t.DUDU, 10.j, 1, 3),
                       (t.MX, t.DUDV,  5.j,-1, 3),
                       (t.MX, t.DUDVb, 3.j, 3, 3),
                       (t.MX, t.DVDV,  2.j, -3, 3),
                       (t.MX, t.DVDVb, 3.j, 1, 3) )

        # And 2nd deriv of W
        w2_terms_e = ( (t.M0, t.DUDU, 4., 0, 4),
                       (t.M0, t.DUDV, 2.,-2, 4),
                       (t.M0, t.DUDVb,2., 2, 4),
                       (t.M0, t.DVDV  ,1.,-4, 4),
                       (t.M0, t.DVDVb ,1., 0, 4),
                       (t.M0, t.DVbDVb,1., 4, 4),
                       (t.MR, t.DUDU, 4., 0, 6),
                       (t.MR, t.DUDV, 2.,-2, 6),
                       (t.MR, t.DUDVb,2., 2, 6),
                       (t.MR, t.DVDV  ,1.,-4, 6),
                       (t.MR, t.DVDVb ,1., 0, 6),
                       (t.MR, t.DVbDVb,1., 4, 6),
                       (t.MC, t.DUDU, 4., 0, 8),
                       (t.MC, t.DUDV, 2.,-2, 8),
                       (t.MC, t.DUDVb,2., 2, 8),
                       (t.MC, t.DVDV  ,1.,-4, 8),
                       (t.MC, t.DVDVb ,1., 0, 8),
                       (t.MC, t.DVbDVb,1., 4, 8),
                       (t.ME, t.DUDU, 4., 2, 6),
                       (t.ME, t.DUDV, 2., 0, 6),
                       (t.ME, t.DUDVb,2., 4, 6),
                       (t.ME, t.DVDV  ,1.,-2, 6),
                       (t.ME, t.DVDVb ,1., 2, 6),
                       (t.ME, t.DVbDVb,1., 6, 6) )

        w2_terms_o = ( (t.MX, t.DUDU,  4.j, 1, 5),
                       (t.MX, t.DUDV,  2.j,-1, 5),
                       (t.MX, t.DUDVb, 2.j, 3, 5),
                       (t.MX, t.DVDV,  1.j,-3, 5),
                       (t.MX, t.DVDVb, 1.j, 1, 5),
                       (t.MX, t.DVbDVb,1.j, 5, 5) )
        
        # Make empty arrays for evens, odds
        even_out = np.zeros( (t.NE, t.ND), dtype=np.complex64)
        odd_out  = np.zeros( (t.NO, t.ND), dtype=np.complex64)

        # Calculate w moments
        self._set_k()
        mask = self.weight.mask
        for wt, terms_e, terms_o in ( (self.weight.w[mask], w0_terms_e, w0_terms_o),
                                      (self.weight.dw[mask], w1_terms_e, w1_terms_o),
                                      (self.weight.d2w[mask], w2_terms_e, w2_terms_o)):
            # What k multipoles are needed for the even moments?
            m_max = np.max(np.abs([term[3] for term in terms_e]))
            N_max = np.max(np.abs([term[4] for term in terms_e]))
            mN = np.zeros( (m_max//2+1, N_max//2+1), dtype=np.complex64)

            # Calculate monopoles first
            kval = self._get_shifted(dx, dy)
            kprod = kval.real * wt
            m = 0
            N = 0
            summand = np.array(kprod)
            mN[m//2,N//2] = np.sum(summand)
            for N in range(2,N_max+1,2):
                summand *= self._ksq
                mN[m//2,N//2] = np.sum(summand)

            # Now do (complex) higher m values
            for m in range(2, m_max+1, 2):
                kprod = kprod * self._kz * self._kz
                N = m
                summand = np.array(kprod)
                mN[m//2, N//2] = np.sum(summand)
                for N in range(m+2,N_max+1,2):
                    summand *= self._ksq
                    mN[m//2,N//2] = np.sum(summand)
                
            # Add all desired terms into output
            for term in terms_e:
                if term[3] >= 0:
                    even_out[term[0],term[1]] += term[2] * mN[ term[3]//2, term[4]//2 ]
                else:
                    # Negative m's must conjugate the multipole
                    even_out[term[0],term[1]] += term[2] * np.conj(mN[ (-term[3])//2, term[4]//2 ])

            # Now move on to the odd-parity multipoles
            m_max = np.max(np.abs([term[3] for term in terms_o]))
            N_max = np.max(np.abs([term[4] for term in terms_o]))
            mN = np.zeros( ( (m_max+1)//2, (N_max+1)//2), dtype=np.complex64)

            # Start at m=1 and move up.  Note a factor j needs to be added to imag part
            kprod = kval.imag * wt * self._kz
            m = 1
            N = 1
            summand = np.array(kprod)
            mN[m//2,N//2] = 1j * np.sum(summand) # Put back 1j
            for N in range(m+2,N_max+1,2):
                summand *= self._ksq
                mN[m//2,N//2] = 1j * np.sum(summand)

            # Now do higher m values
            for m in range(3, m_max+1, 2):
                kprod = kprod * self._kz * self._kz
                N = m
                summand = np.array(kprod)
                mN[m//2, N//2] = np.sum(summand) * 1j
                for N in range(m+2,N_max+1,2):
                    summand *= self._ksq
                    mN[m//2,N//2] = np.sum(summand) * 1j

            # Add all desired terms into output
            for term in terms_o:
                if term[3] >= 0:
                    result = mN[term[3]//2, term[4]//2]
                else:
                    # Negative m's must flip sign of resulting real part
                    z =  mN[(-term[3])//2, term[4]//2]
                    result = -z.real + 1j*z.imag
                odd_out[term[0],term[1]] += term[2] * result

        # Done!
        return Template(self.id, even_out, odd_out, nda=self.nda)
    
    def xy_moment(self,dx):
        '''Interface to fsolve to return x,y moments given input origin shift
        '''
        return self.get_moment(dx[0],dx[1]).odd

    def xy_jacobian(self,dx):
        '''Function to return Jacobian of X & Y moments with respect to
        origin shift dx, for use in solver.  Use the fact that posn derivatives
        of the first moments are the second moments.
        '''
        m = self.get_moment(dx[0],dx[1])
        # ??? It's a bit wasteful that we call get_moments once in xy_moment and again here.
        e = m.even
        return -0.5 * np.array( [ [e[m.MR]+e[m.M1], e[m.M2] ],
                                  [e[m.M2], e[m.MR]-e[m.M1] ] ])

    def recenter(self):
        '''Find dx, dy that null the X and Y moments.  Return
        them and apply the phase shifts to the stored data.
        Returns:
        dx    Shift of origin applied
        error True if there was a failure to converge
        msg   message string on failure
        '''
        dx = np.zeros(2, dtype=float)
        dx, junk, ier, msg = fsolve(self.xy_moment,dx,fprime=self.xy_jacobian, full_output=True,maxfev=500)
        self._set_shifted(dx[0], dx[1])
        
        # Temporary fix to avoid failure when using KBlackmanHarris
        try:
            threshold = np.sqrt(2.0) * np.sqrt(2.0*self.weight.sigsq2)
        except:
            sigma=0.55
            sigsq2 = 0.5*sigma*sigma
            threshold = np.sqrt(2.0) * np.sqrt(2.0*sigsq2)
        
        wandered_too_far = np.abs(dx) >= threshold
        badcentering = wandered_too_far[0] or wandered_too_far[1] or ier <=0
        if (wandered_too_far[0] or wandered_too_far[1]):
            msg+=" but wandered too far"
        return dx, badcentering, msg

    def make_templates(self, sigma_xy, sigma_flux=1., sn_min=0., sigma_max=6.5, sigma_step=1., xy_max=2.,
                           **kwargs):
        ''' Return a list of Template instances that move the object on a grid of
        coordinate origins that keep chisq contribution of flux and center below
        the allowed max.
        sigma_xy    Measurement error on target x & y moments (assumed equal, diagonal)
        sigma_flux  Measurement error on target flux moment
        sn_min      S/N for minimum flux cut applied to targets
        sigma_max   Maximum number of std deviations away from target that template will be used
        sigma_step  Max spacing between shifted templates, in units of measurement sigmas
        xy_max      Max allowed centroid shift, in sky units (prevents runaways)
        '''
        xyshift, error, msg = self.recenter()
        if error:
            return None, "Center wandered too far from starting guess or failed to converge"
        # Determine derivatives of 1st moments on 2 principal axes,
        # and steps will be taken along these grid axes.
        jacobian0 = self.xy_jacobian(np.zeros(2))
        eval, evec = np.linalg.eigh(jacobian0)
        if np.any(eval>=0.):
            return None, "Template galaxy center is not at a flux maximum"

        detj0 = np.linalg.det(jacobian0) # Determinant of Jacobian
        xy_step = np.abs(sigma_step * sigma_xy / eval)
        da = xy_step[0] * xy_step[1]

        # Offset the xy grid by random phase in the grid
        xy_offset = np.random.random(2) - 0.5

        # Now explore contiguous region of xy grid that yields useful templates.
        result = []
        grid_try = set( ( (0,0),) )  # Set of all grid points remaining to try
        grid_done = set()           # Grid points already investigated

        flux_min = sn_min * sigma_flux
        
        while len(grid_try)>0:
            # Try a new grid point
            mn = grid_try.pop()
            grid_done.add(mn)  # Mark it as already attempted
            xy = np.dot(evec, xy_step*(np.array(mn) + xy_offset))  # Offset and scale
            # Ignore if we have wandered too far
            if np.dot(xy,xy) > xy_max*xy_max:
                continue
            m = self.get_moment(xy[0], xy[1])
            e = m.even
            detj = 0.25 * ( e[m.MR]**2-e[m.M1]**2 - e[m.M2]**2)
            # Ignore if determinant of Jacobian has gone negative, meaning
            # we have crossed out of convex region for flux
            if detj <= 0.:
                continue

            # Accumulate chisq that this template would have for a target
            # First: any target will have zero MX, MY
            chisq = (m.odd[m.MX]**2 + m.odd[m.MY]**2) / sigma_xy**2
            # Second: there is suppression by jacobian of determinant
            chisq += -2. * np.log(detj/detj0)
            # Third: target flux will never be below flux_min
            if (e[m.M0] < flux_min):
                chisq += ((flux_min -e[m.M0])/sigma_flux)**2
            if chisq <= sigma_max*sigma_max:
                # This is a useful template!  Add it to output list
                tmpl = self.get_template(xy[0],xy[1])
                tmpl.nda = tmpl.nda * da
                tmpl.jSuppression = detj / detj0
                result.append(tmpl)
                # Try all neighboring grid points not yet tried
                for mn_new in ( (mn[0]+1,mn[1]),
                                (mn[0]-1,mn[1]),
                                (mn[0],mn[1]+1),
                                (mn[0],mn[1]-1)):
                    if mn_new not in grid_done:
                        grid_try.add(mn_new)
        if len(result)==0:
            result.append(None)
            result.append("no templates made")
        return result
            
        

def xyWin(psf, sigma, nominal=None, nIter=3):
    '''Function to use SExtractor's Gaussian-windowed-centroid algorithm
    to determine the center of the (low-noise) image.
    Parameters:
    `psf`:   Image of a source, usually a psf model
    `sigma`: Sigma of the Gaussian weight, in pixel units
    `nominal`: Starting center coordinate.  Defaults to N//2
    `nIter`: Number of rounds of iteration to execute
    Returns:
    (y, x) pair giving center in pixel coordinates (relative to nominal,
             if nominal was given)'''

    # Start at the middle of the image
    if nominal is None:
        center =  np.array([psf.shape[0]//2, psf.shape[1]//2], dtype=float)
    else:
        center = np.array(nominal, dtype=float)

    xx = np.arange(psf.shape[0])
    yy = np.arange(psf.shape[1])

    for i in range(nIter):
        y,x = np.meshgrid( (xx-center[0])/sigma,(yy-center[1])/sigma, indexing='ij')
        # Do this operation in pixel coordinates like SExtractor did
        wt = np.exp(-0.5*(x*x+y*y))
        dyx = np.array( [np.sum(psf*wt*y), np.sum(psf*wt*x)])
        d2yx = np.array( [ [np.sum(psf*wt*(y*y-1)), np.sum(psf*wt*x*y)],
                           [np.sum(psf*wt*x*y), np.sum(psf*wt*(x*x-1))] ] )
        shift = np.linalg.inv(d2yx) @ dyx
        center -= shift * sigma  # Put the sigma back in here

    if nominal is None:
        return center
    else:
        return center - np.array(nominal)
 

            

def simpleImage(image, origin, psf, pixel_scale=1.0, pad_factor=1,
                pixel_noise=None, wcs=None,band=None,
                psf_recenter_sigma=0.,weightSigma=0.65):
    '''Create PSF-corrected k-space image and variance array
    image  postage stamp to use, 2d numpy array, in units of FLUX PER PIXEL.
           2nd axis taken as x.
    origin Location of the origin of the galaxy (2 element array).  Should be near
           the image center. If a wcs is given, these are in the world coordinates
           of the WCS.  If wcs is None, these are 0-index pixel coordinates.
    psf    matching-size postage stamp for the PSF.  We will fix normalization.
           Assumed to have origin at [N//2,N//2] pixel unless `psf_recenter_sigma>0.`
    pixel_scale is number of sky units per pixel, only used if wcs is None
    pad_factor is (integer) zero-padding factor for array before FFT's
    pixel_noise RMS noise of the image pixel values.  
    wcs    A WCS instance giving the map from 0-indexed pixel coordinates to a world
           (sky) system.
    band   imaging band of data given as string, e.g. 'r'
    psf_recenter_sigma:  The sigma (in *pixel* units) of the circular Gaussian window to be
           used to recenter the PSF image, as in SExtractor X/YWIN_IMAGE.  If <=0,
           star is assumed to already be centered in its stamp.
    Returns KData instance.  All quantities therein are in sky units and the flux units of
           the input image.
    '''

    # Find the PSF center, relative to N//2 of its stamp
    psf_shift = np.zeros(2,dtype=float)   # Shift of star center relative to N//2,N//2
    if psf_recenter_sigma > 0:
        nominal = np.array(psf.shape) // 2
        psf_shift = xyWin(psf, sigma=psf_recenter_sigma, nominal=nominal)

    # ??? Do zero padding to some nominal size(s) ?
    N = image.shape[0]
    if not N%2==0:
        raise ValueError('Image and PSF need to be even-dimensioned.')
    ipf = int(pad_factor)
    if pad_factor < 1 or ipf != pad_factor:
        raise ValueError('pad_factor must take positive integer value')

    origin = np.array(origin)
    # Do zero padding if requested
    if ipf > 1:
        Norig=N
        Nnew=N*ipf
        Onew=Nnew//2
        llimnew=Onew-Norig//2
        hlimnew=Onew+Norig//2
        imnew=np.zeros([Nnew,Nnew],dtype=float)
        psfnew=np.zeros([Nnew,Nnew],dtype=float)
        imnew[llimnew:hlimnew,llimnew:hlimnew]=image
        psfnew[llimnew:hlimnew,llimnew:hlimnew]=psf
        image=imnew
        try:
            psf=psfnew
        except:
            pass
        # Shift pixel coords of origin or WCS:
        if wcs is None:
            origin += float(llimnew)
        else:
            # ??? This should be more general than for our WCS class. 
            wcs=copy.deepcopy(wcs)
            wcs.xy0=wcs.xy0 + float(llimnew)
        N=Nnew

    # Make the kx and ky arrays for unit pixel scale
    if numpy_version >= 8:
        ky = np.ones((N,N//2+1), dtype=float) \
             * np.fft.fftfreq(N)[:,np.newaxis]*2.0*np.pi
        kx = np.ones((N,N//2+1), dtype=float) \
             * np.fft.rfftfreq(N)*2.0*np.pi
    else:
        ky = np.ones((N,N//2+1), dtype=float) \
             * np.fft.fftfreq(N)[:,np.newaxis]*2.0*np.pi
        kxtmp = np.fft.fftfreq(N)[0:N//2+1]
        kxtmp[-1] *= -1
        kx = np.ones((N,N//2+1), dtype=float) \
             * kxtmp*2.0*np.pi
    # K area per sample
    d2k = (kx[0,1] - kx[0,0])**2
    
    # Adjust kx, ky, d2k for coordinate mapping
    # and calculate (sky coordinate) displacement of
    # galaxy origin from FFT phase center
    if wcs is None:
        dxy = origin - N//2  # origin was given in pixels
        kx /= pixel_scale
        ky /= pixel_scale
        d2k /= pixel_scale**2
    else:
        # Give needed origin shift in sky units
        dxy = origin - wcs.getuv(np.array([-N//2, -N//2],dtype=float))
        # Put k values into an array of shape (2, N^2)
        kxy = np.vstack((kx.flatten(),ky.flatten()))
        # Transform k's and put back into shape of the k arrays
        kxy = wcs.getuv_k(kxy)
        kx = kxy[0].reshape(kx.shape)
        ky = kxy[1].reshape(ky.shape)
        
        # Rescale pixel area in k space, need absolute value of determinant 
        # to keep d2k positive
        d2k /= np.abs(wcs.getdet())
    
    kmax = 1.07635*np.pi/weightSigma # wt(kr) >= kmax is 0
        
    kymax = np.max([np.argmin(np.abs(ky[:N//2,0]-kmax))+1,
                   np.argmin(np.abs(ky[:N//2,0]+kmax))+1]) # index of maximum considered ky (+ 1 to avoid rounding issues)
    kxmax = np.max([np.argmin(np.abs(kx[0,:N//2]-kmax))+1,
                   np.argmin(np.abs(kx[0,:N//2]+kmax))+1]) # index of maximum considered kx (+ 1 to avoid rounding issues)
    
    kx = np.vstack([kx[:kxmax,:kxmax],kx[-kxmax:,:kxmax]]) # cropping kx
    ky = np.vstack([ky[:kymax,:kymax],ky[-kymax:,:kymax]]) # cropping ky
        
    kval =  np.fft.rfft2(image)
    
    kval = np.vstack([kval[:kxmax,:kymax],kval[-kxmax:,:kymax]]) # cropping kval
    
    # Flip signs to shift coord origin from [0,0] to N/2,N/2
    kval[1::2,::2] *= -1.
    kval[::2,1::2] *=-1.

    # Process and correct for the PSF
    if psf is not None:
        kpsf  =  np.fft.rfft2(psf)
        
        kpsf = np.vstack([kpsf[:kxmax,:kymax],kpsf[-kxmax:,:kymax]]) # cropping kpsf
        
        kpsf[1::2,::2] *= -1.
        kpsf[::2,1::2] *=-1.
        # Normalize PSF to unit flux (note DC is at [0,0])
        kpsf /= kpsf[0,0]
        if np.any(psf_shift):
            # Adjust PSF phases to center it
            phase = kx * psf_shift[1] + ky * psf_shift[0]  # ??? coord swap?
            kpsf =  np.exp(1j*phase) * kpsf
        
        # Correct for PSF
        kval /= kpsf
    
    
    # Double the weight on samples whose conjugates are missing
    conjugate = np.zeros_like(kval, dtype=bool)
    conjugate[:,1:N//2] = True

    # Make variance array if we have noise
    kvar = None
    if pixel_noise is not None:
        kvar = np.ones_like(kval,dtype=float) * (pixel_noise*N)**2
        if psf is not None:
            kvar /= (kpsf.real*kpsf.real + kpsf.imag*kpsf.imag)

    # Apply phase shift to move center
    phase = kx * dxy[0] + ky * dxy[1]
    kval *=np.exp(1j*phase)

    if psf_recenter_sigma > 0:
        return KData(kval, kx, ky, d2k, conjugate, kvar,band),psf_shift
    else:
        return KData(kval, kx, ky, d2k, conjugate, kvar,band), None
        

def simpleImageCross(image1, image2, origin1, origin2, psf1, psf2, pixel_scale=1.0, pad_factor=1, wcs1=None, wcs2=None, band=None):
    '''Create PSF-corrected k-space image and cross-variance array for two image
    image1 & image2  postage stamps to use, 2d numpy array, in units of FLUX PER PIXEL.
                     2nd axis taken as x.
    origin1 & origin2 Location of the origin of the galaxy (2 element array).  Should be near
                      the image center. If a wcs is given, these are in the world coordinates
                      of the WCS.  If wcs is None, these are 0-index pixel coordinates.
    psf1 & psf2       matching-size postage stamp for the PSF.  We will fix normalization.
                      Assumed to have origin at [N/2,N/2] pixel.
    pixel_scale is number of sky units per pixel, only used if wcs is None
    wcs1 & wcs2    A WCS instance giving the map from 0-indexed pixel coordinates to a world
                   (sky) system.
    band           imaging band of data given as string, e.g. 'r'

    Returns KData instance.  All quantities therein are in sky units and the flux units of
           the input image.
    '''

    if psf1.shape != image1.shape or psf1.ndim != 2 or image1.shape[0]!=image1.shape[1]:
        raise Exception('PSF1 and image1 shape must be matching square arrays for simpleImageCross')

    if psf2.shape != image2.shape or psf2.ndim != 2 or image2.shape[0]!=image2.shape[1]:
        raise Exception('PSF2 and image2 shape must be matching square arrays for simpleImageCross')
    # ??? Do zero padding to some nominal size(s)

    N1 = image1.shape[0]
    N2 = image2.shape[0]
    if not N1%2==0:
        raise Exception('Image1 and PSF1 need to be even-dimensioned.')

    if not N2%2==0:
        raise Exception('Image1 and PSF1 need to be even-dimensioned.')

    if not N1==N2:
        raise Exception('Image1 and Image2 need to have same dimensions.')
    N=N1

    if pad_factor > 1:
        Norig=N
        Nnew =N*pad_factor
        Onew =Nnew/2.
        llimnew=int(Onew-Norig/2.)
        hlimnew=int(Onew+Norig/2.)

        im1new=np.zeros([Nnew,Nnew])
        psf1new=np.zeros([Nnew,Nnew])
        im2new=np.zeros([Nnew,Nnew])
        psf2new=np.zeros([Nnew,Nnew])

        im1new[llimnew:hlimnew,llimnew:hlimnew]=image1
        psf1new[llimnew:hlimnew,llimnew:hlimnew]=psf1
        im2new[llimnew:hlimnew,llimnew:hlimnew]=image2
        psf2new[llimnew:hlimnew,llimnew:hlimnew]=psf2

        image1=im1new
        psf1=psf1new
        image2=im2new
        psf2=psf2new

        if wcs1 is None:
            origin=np.array([Onew,Onew])
        else:
            wcs1=copy.deepcopy(wcs1)
            wcs1.xy0=wcs1.xy0-(Norig/2.)+(Nnew/2.)

        if wcs2 is None:
            origin=np.array([Onew,Onew])
        else:
            wcs2=copy.deepcopy(wcs2)
            wcs2.xy0=wcs2.xy0-(Norig/2.)+(Nnew/2.)

        N=Nnew

    kval1 =  np.fft.rfft2(image1)
    # Flip signs to shift coord origin from [0,0] to N/2,N/2
    kval1[1::2,::2] *= -1.
    kval1[::2,1::2] *=-1.

    kval2 =  np.fft.rfft2(image2)
    # Flip signs to shift coord origin from [0,0] to N/2,N/2
    kval2[1::2,::2] *= -1.
    kval2[::2,1::2] *=-1.

    # Same for PSF
    kpsf1  =  np.fft.rfft2(psf1)
    kpsf1[1::2,::2] *= -1.
    kpsf1[::2,1::2] *=-1.

    kpsf2  =  np.fft.rfft2(psf2)
    kpsf2[1::2,::2] *= -1.
    kpsf2[::2,1::2] *=-1.

    # Normalize PSF to unit flux (note DC is at [0,0])
    kpsf1 /= kpsf1[0,0]
    kpsf2 /= kpsf2[0,0]

    # Correct for PSF
    kval1 /= kpsf1
    kval2 /= kpsf2
    
    # Double the weight on samples whose conjugates are missing
    conjugate = np.zeros_like(kpsf1, dtype=bool)
    conjugate[:,1:N/2] = True


    # Make the kx and ky arrays for unit pixel scale
    if numpy_version >= 8:
        ky1 = np.ones_like(kpsf1, dtype=float) \
            * np.fft.fftfreq(N)[:,np.newaxis]*2.0*np.pi
        kx1 = np.ones_like(kpsf1, dtype=float) \
            * np.fft.rfftfreq(N)*2.0*np.pi

        ky2 = np.ones_like(kpsf2, dtype=float) \
            * np.fft.fftfreq(N)[:,np.newaxis]*2.0*np.pi
        kx2 = np.ones_like(kpsf2, dtype=float) \
            * np.fft.rfftfreq(N)*2.0*np.pi
    else:
        ky1 = np.ones_like(kpsf1, dtype=float) \
            * np.fft.fftfreq(N)[:,np.newaxis]*2.0*np.pi
        kx1tmp = np.fft.fftfreq(N)[0:N/2+1]
        kx1tmp[-1] *= -1
        kx1 = np.ones_like(kpsf1, dtype=float) \
            * kx1tmp*2.0*np.pi

        ky2 = np.ones_like(kpsf2, dtype=float) \
            * np.fft.fftfreq(N)[:,np.newaxis]*2.0*np.pi
        kx2tmp = np.fft.fftfreq(N)[0:N/2+1]
        kx2tmp[-1] *= -1
        kx2 = np.ones_like(kpsf2, dtype=float) \
            * kx2tmp*2.0*np.pi
    # K area per sample
    d2k1 = (kx1[0,1] - kx1[0,0])**2
    d2k2 = (kx2[0,1] - kx2[0,0])**2

    # Adjust kx, ky, d2k for coordinate mapping
    # and calculate (sky coordinate) displacement of
    # galaxy origin from FFT phase center

    if wcs1 is None:
        dxy1 = np.array(origin1) - N/2  # origin was given in pixels
        kx1 /= pixel_scale
        ky1 /= pixel_scale
        d2k1 /= pixel_scale**2
        dxy2 = np.array(origin2) - N/2  # origin was given in pixels
        kx2 /= pixel_scale
        ky2 /= pixel_scale
        d2k2 /= pixel_scale**2
    else:
        # Give needed origin shift in sky units
        dxy1 = np.array(origin1) - wcs1.getuv( np.array([-N/2, -N/2],dtype=float))
        # Put k values into an array of shape (2, N^2)
        kxy1 = np.vstack((kx1.flatten(),ky1.flatten()))
        # Transform k's and put back into shape of the k arrays
        kxy1 = wcs1.getuv_k(kxy1)
        kx1 = kxy1[0].reshape(kx1.shape)
        ky1 = kxy1[1].reshape(ky1.shape)
        
        # Rescale pixel area in k space, need absolute value of determinant 
        # to keep d2k positive
        d2k1 /= np.abs(wcs1.getdet())

        # Give needed origin shift in sky units
        dxy2 = np.array(origin2) - wcs2.getuv( np.array([-N/2, -N/2],dtype=float))
        # Put k values into an array of shape (2, N^2)
        kxy2 = np.vstack((kx2.flatten(),ky2.flatten()))
        # Transform k's and put back into shape of the k arrays
        kxy2 = wcs2.getuv_k(kxy2)
        kx2 = kxy2[0].reshape(kx2.shape)
        ky2 = kxy2[1].reshape(ky2.shape)
        
        # Rescale pixel area in k space, need absolute value of determinant 
        # to keep d2k positive
        d2k2 /= np.abs(wcs2.getdet())

    # Apply phase shift to move center
    phase1 = kx1 * dxy1[0] + ky1 * dxy1[1]
    kval1 *=np.exp(1j*phase1)
    phase2 = kx2 * dxy2[0] + ky2 * dxy2[1]
    kval2 *=np.exp(1j*phase2)

    if wcs2 is not None:
        dx12=kx1-kx2
        dy12=ky1-ky2
        phase12=kx2*dx12 + ky2*dy12
        kval2 *=np.exp(1j*phase12)
    
    # Make cross-variance array if we have noise
    kvar = kval1 * np.conj(kval2)
    kvar *= pad_factor*pad_factor
    kvar = kvar.real



class WCS(object):
    '''A simple WCS, which will be an affine transformation from pixel coords
    xy to some sky coordinates uv.'''
    def __init__(self, duv_dxy, xyref=[0,0],uvref=[0,0]):
        ''' Define an affine transformation from pixel to sky coordinates
        (uv - uvref) = duv_dxy * (xy - xyref)
        -----
        parameters

        duv_dxy:   The 2x2 Jacobian matrix such that duv_dxy[i,j] = du_i / dx_j
        xyref:     2-element pixel coordinates of reference point
        uvref:     2-element sky coordinates of reference point
        '''
        self.jac=duv_dxy
        self.jacinv = np.linalg.inv(self.jac)
        self.xy0=np.array(xyref)
        self.uv0=np.array(uvref)
    
        return
        
    def getxy(self,uv):
        # Map from uv to xy.  Input can be of shape (2) or (2,N)
        return np.dot(self.jacinv, (uv - self.uv0)) + self.xy0

    def getuv(self,xy):
        # Map from xy to uv.  Input can be of shape (2) or (2,N)
        return np.dot(self.jac,(xy - self.xy0)) + self.uv0

    def getuv_k(self, kxy):
        # Map from k_x, k_y to k_u, k_v.  Input can be of shape (2) or (2,N)
        return np.dot(self.jacinv.T, kxy)

    def getdet(self):
        # Get determinant | duv / dxy |
        return self.jac[0,0]*self.jac[1,1] - self.jac[0,1]*self.jac[1,0]
    

    
