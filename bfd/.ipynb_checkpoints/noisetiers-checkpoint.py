# Classes and functions for defining and using noise tiers,
# and for building selection PQR's that are functions of
# the covariance matrix elements.

import sys
import numpy as np
from astropy import version 
import astropy.io.fits as fits
from .momentcalc import *
from .keywords import *
import re

# We'll want to look at the covariance matrices in a few different ways.

class MultipoleCov:
    def __init__(self,cov=None):
        '''Create multipole representation of the even-parity covariance matrix.
        Input covariance matrix is either a 2d full cov matrix or 1d
        packed version of it.'''
        self.mono = np.zeros(5,dtype=float)
        self.e1 = np.zeros(3,dtype=float)
        self.e2 = np.zeros(3,dtype=float)
        if cov is None:
            self.q1 = 0.
            self.q2 = 0.
            return
        # create values from the input covariance
        m = Moment()

        if type(cov)==MomentCovariance:
            mono, e1, e2, q1, q2 =  self.bulkTransform(cov.even.reshape(1,m.NE,m.NE))
        elif cov.ndim==2:
            mono, e1, e2, q1, q2 =  self.bulkTransform(cov.reshape(1,m.NE,m.NE))
        elif cov.ndim==1:
            mono, e1, e2, q1, q2 =  self.bulkTransform(MomentCovariance.unpack(cov))
        else:
            raise ValueError('Wrong dimensionality for input covariance')
            
        self.mono = mono.squeeze()
        self.e1 = e1.squeeze()
        self.e2 = e2.squeeze()
        self.q1 = q1[0]
        self.q2 = q2[0]

    @classmethod
    def bulkTransform(cls,covs):
        '''Convert an input array of packed covariances into
        Nx6, Nx3, Nx3, N, N arrays of multipole elements
        that would be mono, e1, e2, q1,q2.
        Input array is either (N,5,5) or (N,15) shape.'''
        m= Moment()
        if covs.ndim==2:
            cov2d = MomentCovariance.bulkUnpack(covs)
        elif covs.ndim==3:
            cov2d = covs
        else:
            raise ValueError('Input covs array must have dimension 2 or 3')
        npts = cov2d.shape[0]
        mono = np.zeros((npts,5),dtype=float)
        e1 = np.zeros((npts,3),dtype=float)
        e2 = np.zeros((npts,3),dtype=float)

        # First, elements with m=0 symmetry, radial powers 0,2,4,6,8 in k
        mono[:,0] = cov2d[:,m.M0,m.M0]
        mono[:,1] = cov2d[:,m.M0,m.MR]
        mono[:,2] = cov2d[:,m.M0,m.MC]
        mono[:,3] = cov2d[:,m.MR,m.MC]
        mono[:,4] = cov2d[:,m.MC,m.MC]
        # quadrupole moments at different radial orders 2, 4, 6
        e1[:,0] = cov2d[:,m.M0,m.M1]/mono[:,1]
        e2[:,0] = cov2d[:,m.M0,m.M2]/mono[:,1]
        e1[:,1] = cov2d[:,m.MR,m.M1]/mono[:,2]
        e2[:,1] = cov2d[:,m.MR,m.M2]/mono[:,2]
        e1[:,2] = cov2d[:,m.MC,m.M1]/mono[:,3]
        e2[:,2] = cov2d[:,m.MC,m.M2]/mono[:,3]
        # m=4 at radial order 4 in k
        q1 = (cov2d[:,m.M1,m.M1]-cov2d[:,m.M2,m.M2])/mono[:,2]
        q2 = 2*cov2d[:,m.M1,m.M2]/mono[:,2]
        
        # Take log of monopoles
        mono = np.log(mono)
        # And make k>0 relative to k=0
        mono[:,1:] -= mono[:,0][:,np.newaxis]

        return mono, e1, e2, q1, q2

    def isotropize(self):
        # Make the covariance matrix rotationally invariant
        self.q1 = self.q2 = 0
        self.e1.fill(0.)
        self.e2.fill(0.)
        return

    def matrices(self):
        # Return a MomentCovariance object specifed by these multipoles
        m = Moment()
        even = np.zeros((m.NE,m.NE),dtype=float)
        odd = np.zeros((m.NO,m.NO), dtype=float)
        cc = np.exp(self.mono)
        cc[1:] *= cc[0]  # Scale k>0 by by k=0
        even[m.M0, m.M0] = cc[0]  # k^0
        even[m.M0, m.MR] = cc[1]  # k^2
        even[m.MR, m.M0] = cc[1]  # 
        even[m.MR, m.MR] = cc[2]  # k^4
        even[m.M0, m.MC] = cc[2]
        even[m.MC, m.M0] = cc[2]
        even[m.MR, m.MC] = cc[3]  # k^6
        even[m.MC, m.MR] = cc[3]  
        even[m.MC, m.MC] = cc[4]  # k^8

        odd[m.MX, m.MX] = 0.5 * cc[1] * (1+self.e1[0])
        odd[m.MY, m.MY] = 0.5 * cc[1] * (1-self.e1[0])
        odd[m.MX, m.MY] = 0.5 * cc[1] * self.e2[0]
        odd[m.MY, m.MX] = 0.5 * cc[1] * self.e2[0]

        even[m.M0, m.M1] = cc[1] * self.e1[0]
        even[m.M0, m.M2] = cc[1] * self.e2[0]
        even[m.MR, m.M1] = cc[2] * self.e1[1]
        even[m.MR, m.M2] = cc[2] * self.e2[1]
        even[m.MC, m.M1] = cc[3] * self.e1[2]
        even[m.MC, m.M2] = cc[3] * self.e2[2]
        # transposes
        even[m.M1, m.M0] = even[m.M0, m.M1]
        even[m.M2, m.M0] = even[m.M0, m.M2]
        even[m.M1, m.MR] = even[m.MR, m.M1]
        even[m.M2, m.MR] = even[m.MR, m.M2]
        even[m.M1, m.MC] = even[m.MC, m.M1]
        even[m.M2, m.MC] = even[m.MC, m.M2]

        even[m.M1, m.M1] = 0.5 * cc[2] * (1+self.q1)
        even[m.M2, m.M2] = 0.5 * cc[2] * (1-self.q1)
        even[m.M1, m.M2] = 0.5 * cc[2] * self.q2
        even[m.M2, m.M1] = 0.5 * cc[2] * self.q2
        
        return MomentCovariance(even, odd)

class CovarianceCompressor:
    def __init__(self, covs=None):
        '''Class that will be a compressor / decompressor for
        MomentCovariance matrices.  The 14 DOF of a matrix will
        be reduced to functions of just 4 arguments: 
        * a0 = log(Cff) = flux noise
        * a2 = log(Cov(MF,MX) / Cff) ~ PSF size
        * e1, e2 from (Cxx-Cyy)/(Cxx+Cyy) and 2Cxy/(.)

        * Relations between these and the other mutipole
        * coefficients are fit to the input array covs1d.
        '''

        if covs is None:
            # Create an empty shell object
            return

        mono, e1, e2, q1, q2 = MultipoleCov.bulkTransform(covs)

        # Use k^2 monopole relative to k0 as a param
        a2 = mono[:,1]
        # Regress other monopoles against a2
        A = np.vstack((np.ones_like(a2),a2)).T
        ainv = np.linalg.inv(np.matmul(A.T,A))
        # Get slope and intercept of each monopole coeff vs log(a2)
        self.mono_vs_a2 = np.zeros((mono.shape[1]-1,2),dtype=float)
        for i in range(mono.shape[1]-1):
            self.mono_vs_a2[i,:] = np.dot(ainv, np.dot(A.T, mono[:,i+1]))

        # Get relation of higher-order e's to lower-order ones
        # as a multiplicative factor.
        ee = np.concatenate( (e1, e2), axis=0)
        e0 = ee[:,0]
        self.e_vs_e2 = np.dot(ee.T,e0) / np.dot(e0,e0)
        
        # save the range of applicability
        self.range_min = np.min(mono,axis=0)
        self.range_max = np.max(mono,axis=0)

        return

    def compress(self, cov):
        '''Return compressed version of the input cov matrix.
        If input is 2d it's assumed a full matrix, else
        assumed to be packed if 1d.
        Returns 4-element compressed version.
        '''
        mc = MultipoleCov(cov)
        return np.array([mc.mono[0], mc.mono[1], mc.e1[0], mc.e2[0]])

    def bulkCompress(self,covs):
        '''Return Nx4 array of compressed coefficients to represent
        covariances input by either an (N,5,5) or (N,15) array.'''
        mono, e1, e2, _, _ = MultipoleCov.bulkTransform(covs)
        return np.array( [mono[:,0], mono[:,1], e1[:,0], e2[:,0]]).T

    def uncompress(self, a0, a2, e1, e2, q1=0., q2=0.):
        '''Return full MomentCovariance from a compressed representation.'''
        mc = MultipoleCov()
        mc.mono[0] = a0
        mc.mono[1:] = np.dot(self.mono_vs_a2, np.array([1.,a2]))
        mc.e1 = e1 * self.e_vs_e2
        mc.e2 = e2 * self.e_vs_e2
        mc.q1 = q1
        mc.q2 = q2
        return mc.matrices()

    def saveDict(self):
        '''Return a dictionary that specifies the state of compressor'''
        h = {}
        for i in range(1,self.mono_vs_a2.shape[0]):
            for j in range(self.mono_vs_a2.shape[1]):
                h['MONO_{:1d}_{:1d}'.format(i,j)] = self.mono_vs_a2[i,j]
        for i in range(1,self.e_vs_e2.shape[0]):
            h['ELLIP_{:1d}'.format(i)] = self.e_vs_e2[i]
        return h

    @classmethod
    def loadDict(cls,d):
        '''Build an instance from a dict-like structure'''
        mre = re.compile(r'^MONO_(\d)_(\d)')
        ere = re.compile(r'^ELLIP_(\d*)')
        mono = []
        ellip = []
        # Look for useful entries in the dictionary
        for k,v in d.items():
            m = mre.match(k)
            if m:
                # Record a tuple for entry into monopole table
                mono.append( (int(m.group(1)), int(m.group(2)), float(v)))
                continue
            m = ere.match(k)
            if m:
                # Record a tuple for entry into monopole table
                ellip.append( (m.group(1), float(v)))

        if not mono or not ellip:
            # Can't build the object
            return None
        out = cls()
        # Now build arrays
        ij = np.array([ [t[0],t[1]] for t in mono])
        dims = np.max(ij, axis=0) + 1
        out.mono_vs_a2 = np.zeros( (dims[0], dims[1]), dtype=float)
        # The first row of the monopole compressor is an identity transform
        out.mono_vs_a2[0,1] = 1.
        out.mono_vs_a2[ij[:,0],ij[:,1]] = np.array([t[2] for t in mono])

        i = np.array([int(t[0]) for t in ellip])
        dim = np.max(i) + 1
        out.e_vs_e2 = np.zeros(dim,dtype=float)
        # First entry should be unity
        out.e_vs_e2[0] = 1.
        out.e_vs_e2[i] =  np.array([t[1] for t in ellip])

        return out
        
        

    
class NoiseTier:
    def __init__(self,id=0,
                     covNodes=None,    # Array of covs for selection-function interp
                     nodeIDs=None,     # index of what node each cov belongs to.  0 is nominal
                     compressor=None,  # CovarianceCompressor used
                     startA = None,    # array giving origin of the grid in compressed space
                     stepA = None,     # array of step sizes of grid
                     indexA = None,    # array of this tier's position in grid
                     fluxMin = None,   # Lower limit on target flux, if set yet
                     fluxMax = None,   # Upper limit, if there is one
                     wtN = None,       # N parameter for BFD weight function
                     wtSigma = None):   # Sigma of BFD weight
                         
        '''Description of a single noise tier.  Can be save to or retrieved from
        a FITS extension which contains a bunch of header info and also a table 
        of covariance matrices and potentially their selection PQRs.'''
        self.id = id
        self.cc = compressor
        self.fluxMin = fluxMin
        self.fluxMax = fluxMax
        self.wtN = wtN
        self.wtSigma = wtSigma

        self.pqrNodes = None   # Add this to object outside of init
        self.interpolator = None  # Create a PQRInterpolator when needed
        
        if covNodes is None:
            self.covNodes=None
        else:
            self.covNodes = np.array(covNodes)
        if nodeIDs is None:
            self.nodeIDs=None
        else:
            self.nodeIDs = np.array(nodeIDs)
        if startA is None:
            startA=None
        else:
            self.startA = np.array(startA)
        if stepA is None:
            self.stepA = None
        else:
            self.stepA = np.array(stepA)
        if indexA is None:
            self.indexA=None
        else:
            self.indexA = np.array(indexA)
    

    def nominalCov(self):
        if self.covNodes is None or self.nodeIDs is None:
            raise ValueError('No covariance has been set')
        whereIsNominal = np.where(self.nodeIDs==0)[0][0]
        return  MomentCovariance.unpack(self.covNodes[whereIsNominal])

    def _buildInterpolator(self):
        '''Use PQR values at covNodes to set up an interpolator.'''

    def pqrSel(self, covs):
        '''Interpolate deselection PQR to the given array of 
        covariance matrices.  Returns a packed version of pqr.'''
        if self.interpolator is None:
            # Must build an interpolator across the range
            if self.pqrNodes is None:
                raise ValueError('PQR data not available for NoiseTier to create interpolation function')
            self.interpolator = PqrInterpolator(self.nodeIDs,
                                                self.covNodes,
                                                self.pqrNodes,
                                                self.cc)
        # Call the interpolator
        return self.interpolator(covs)
        
    def saveHDU(self):
        '''Save this tier as a FITS extension'''
        h = fits.Header()
        h[hdrkeys['tierNumber']] = self.id

        m = Moment()
         
        cov = self.nominalCov()

        h['COVMXMX'] = cov.odd[m.MX,m.MX]
        h['COVMXMY'] = cov.odd[m.MX,m.MY]
        h['COVMYMY'] = cov.odd[m.MY,m.MY]

        h['SIG_XY'] = np.sqrt(cov.odd[m.MX,m.MX])
        h['SIG_FLUX'] = np.sqrt(cov.even[m.M0,m.M0])
        
        if self.startA is not None:
            # record range of tier in compressed-cov space
            for i in range(self.startA.shape[0]):
                h['STARTA{:02d}'.format(i)] = self.startA[i]
                h['STEPA{:02d}'.format(i)] = self.stepA[i]
                h['INDEXA{:02d}'.format(i)] = self.indexA[i]

        if self.cc is not None:
            for k,v in self.cc.saveDict().items():
                h[k] = v

        # Fluxmin and max
        if self.fluxMin is not None:
            h[hdrkeys['fluxMin']] = self.fluxMin
        if self.fluxMax is not None:
            h[hdrkeys['fluxMax']] = self.fluxMax
        # Weighting parameters
        if self.wtN is not None:
            h[hdrkeys['weightN']] = self.wtN
        if self.wtSigma is not None:
            h[hdrkeys['weightSigma']] = self.wtSigma

        cols = [fits.Column(name=colnames['nodeID'], array=self.nodeIDs,format='J'),
                fits.Column(name=colnames['covariance'], array=self.covNodes,
                            format='{:d}E'.format(self.covNodes.shape[1]))]
        if self.pqrNodes is not None:
            cols.append(fits.Column(name=colnames['pqrSel'], array=self.pqrNodes,
                            format='{:d}E'.format(self.pqr.shape[1])))

        return fits.BinTableHDU.from_columns(cols,
                    header=h, name='TIER{:04d}'.format(self.id))


    @classmethod
    def loadHDU(cls,hdu):
        '''Create a NoiseTier from a FITS HDU'''
        out = cls()
        out.id = hdu.header[hdrkeys['tierNumber']]
        out.covNodes = hdu.data[colnames['covariance']]
        out.nominalCov = MomentCovariance.unpack(out.covNodes[0])
        out.nodeIDs = hdu.data[colnames['nodeID']]
        if colnames['pqrSel'] in hdu.columns.names:
            out.pqrNodes = hdu.data[colnames['pqrSel']]
        else:
            out.pqrNodes = None
        
        # Create a compressor
        out.cc = CovarianceCompressor.loadDict(hdu.header)

        # Get start/stop/index of compressed coords
        startre = re.compile(r'^STARTA(\d*)')
        stepre = re.compile(r'^STEPA(\d*)')
        indexre = re.compile(r'^INDEXA(\d*)')
        start = []
        step = []
        index = []
        # Look for useful entries in the dictionary
        for k,v in hdu.header.items():
            m = startre.match(k)
            if m:
                # Record a tuple for entry into monopole table
                start.append( (int(m.group(1)), float(v)))
                continue
            m = stepre.match(k)
            if m:
                # Record a tuple for entry into monopole table
                step.append( (int(m.group(1)), float(v)))
                continue
            m = indexre.match(k)
            if m:
                # Record a tuple for entry into monopole table
                index.append( (int(m.group(1)), int(v)))
        # Assuming that all three or none are present
        if (start and step and index):
            i = np.array( [t[0] for t in start])
            v = np.array( [t[1] for t in start])
            dim = np.max(i)+1
            out.startA = np.zeros(dim,dtype=float)
            out.stepA = np.zeros(dim,dtype=float)
            out.indexA = np.zeros(dim,dtype=int)
            out.startA[i] = v
            i = np.array([t[0] for t in step])
            v = np.array([t[1] for t in step])
            out.stepA[i] = v
            i = np.array( [t[0] for t in index])
            v = np.array( [t[1] for t in index])
            out.indexA[i] = v
        else:
            out.startA=None
            out.stepA=None
            out.indexA=None

        if hdrkeys['weightN'] in hdu.header:
            out.wtN = hdu.header[hdrkeys['weightN']]
        else:
            out.wtN = None
        if hdrkeys['weightSigma'] in hdu.header:
            out.wtSigma = hdu.header[hdrkeys['weightSigma']]
        else:
            out.wtSigma = None
        if hdrkeys['fluxMin'] in hdu.header:
            out.fluxMin = hdu.header[hdrkeys['fluxMin']]
        else:
            out.fluxMin = None
        if hdrkeys['fluxMax'] in hdu.header:
            out.fluxMax = hdu.header[hdrkeys['fluxMax']]
        else:
            out.fluxMax = None
        
        return out

class TierCollection:
    def __init__(self, covs=None, stepA=[0.2,0.1], minTargets=10,
                     wtN=None, wtSigma=None,
                     snMin=8., snMax=None,
                     fluxMin=None, fluxMax=None,
                     nCovGrid=4):
        '''Class for a collection of noise tiers to use for a survey.
        A compressor is built from the input covariance array.
        Noise tiers are defined by creating cells in the
        space of the first len(stepA) compressed quantities, such that cells 
        have size no larger than stepA.  Noise tiers having
        less than minTargets in them are not kept.

        It is assumed that the compressed covariance has the number of elements
        in stepA, followed by 2 more for e1 and e2 descriptors.

        Each tier is assigned flux min/max as the more constraining of the
        `fluxMin/Max` or the `snMin/Max` as evaluated at nominal flux noise.

        `nCovGrid` gives the number of points across each dimension of the compressed
        covariance space that will be used in making a grid of covariance nodes
        for each tier.
        
        '''
        
        if covs is None:
            # Make an empty shell of class
            return

        m = Moment()
        
        self.wtN = wtN
        self.wtSigma = wtSigma
        self.cc = CovarianceCompressor(covs)
        # Create grid boundaries in compressed space
        self.stepA = np.array(stepA)
        self.dimA = self.stepA.shape[0]
        self.nSteps = np.ceil( (self.cc.range_max-self.cc.range_min)[:self.dimA] / stepA).astype(int)
        # Make the steps v slightly larger than equal division to avoid max value getting
        # pushed past the last bin:
        self.stepA = 1.0001* (self.cc.range_max - self.cc.range_min)[:self.dimA] / self.nSteps
        self.startA = self.cc.range_min[:self.dimA].copy()

        # Assign each target a bin number, and count them
        comp = self.cc.bulkCompress(covs)
        indices = np.floor( (comp[:,:self.dimA] - self.startA) / self.stepA).astype(int)
        # Turn these into indices into a flattened array
        ii = np.ravel_multi_index( [indices[:,k] for k in range(indices.shape[1])], self.nSteps)
        tmpid, count = np.unique(ii, return_counts=True)
        keep = count >= minTargets

        # Create array of assignments of targets to tiers.
        # Default (no assignment) will be -1
        self.assignments = np.ones(covs.shape[0], dtype=int) * -1
        # Create and populate useful tiers
        self.tiers = []
        for i,k in zip(tmpid,keep):
            if not k:
                # Skip underpopulated tiers
                continue

            # Assign a permanent id to this tier
            tierID = len(self.tiers)

            # Calculate mean cov of its member targets
            members = ii==i
            self.assignments[members] = tierID
            meanCov = np.mean(covs[members], axis=0)
            if meanCov.ndim==1:
                # Unpack it
                meanCov = MomentCovariance.unpack(meanCov)
            else:
                # make an object out of it 
                meanCov = MomentCovariance(meanCov)
            # Isotropize
            meanCov = meanCov.isotropize()

            # Set flux limits based on this covariance
            sigFlux = np.sqrt(meanCov.even[m.M0,m.M0])
            if snMin is None:
                fMin = fluxMin
            else:
                fMin = snMin * sigFlux;
                if fluxMin is not None:
                    fMin = max(fMin, fluxMin) # Take more stringent

            if snMax is None:
                fMax = fluxMax
            else:
                fMax = snMax * sigFlux;
                if fluxMax is not None:
                    fMax = min(fMax, fluxMax) # Take more stringent
            
            # Pack the mean covariance and make it the first
            # row of a matrix of them, with ID number 0
            tierCovs = [meanCov.pack()]
            tierCovIDs = [0]

            # What is the location of this tier in compressed space?
            tierIndices = np.unravel_index(i, self.nSteps)
            
            # Now make covariance matrices for the
            # a grid of points across the noise tier region,
            # with each having a nonzero e1 to get slope
            aMin = self.startA + self.stepA*tierIndices
            aMax = aMin +  self.stepA
            aPts = np.linspace(aMin, aMax, nCovGrid).T
            xx = np.array(np.meshgrid(*aPts))

            # *** There is an assumption here that the compressed space
            # *** is the gridded a space augmented by e1 and e2.
            # Step through all combinations of grid points
            for j,aa in enumerate(xx.reshape(self.dimA,-1).T):
                for e1 in (0, 0.05):
                    cov = self.cc.uncompress(*aa, e1, 0.)
                    tierCovs.append(cov.pack())
                    tierCovIDs.append(j+1)

            self.tiers.append(NoiseTier(id=tierID,
                                        covNodes=tierCovs,
                                        nodeIDs = tierCovIDs,
                                        compressor=self.cc,
                                        startA = self.startA,
                                        stepA = self.stepA,
                                        indexA = tierIndices,
                                        wtN = self.wtN,
                                        wtSigma = self.wtSigma,
                                        fluxMin = fMin,
                                        fluxMax = fMax))
            
        return

    def save(self,fitsname):
        ''' Save all noise tier information to the specified FITS file'''
        phdu = fits.PrimaryHDU()
        hdul = fits.HDUList( [phdu] + [i.saveHDU() for i in self.tiers])
        hdul.writeto(fitsname, overwrite=True)
        return

    @classmethod
    def load(cls, fitsname):
        '''Load all tiers' information from the given FITS file'''
        out = cls()
        ff = fits.open(fitsname)
        out.tiers = []
        checkTier = re.compile(r'^TIER(\d*)')
        for hdu in ff:
            if 'EXTNAME' not in hdu.header or not checkTier.match(hdu.header['EXTNAME']):
                # Not a noise tier HDU.  Skip it
                continue
            out.tiers.append(NoiseTier.loadHDU(hdu))

        out.cc = out.tiers[0].cc
        out.stepA = out.tiers[0].stepA
        out.startA = out.tiers[0].startA
        out.dimA = out.startA.shape[0]
        out.wtN = out.tiers[0].wtN
        out.wtSigma = out.tiers[0].wtSigma
        out.nSteps = np.zeros_like(out.startA, dtype=int)
        for t in out.tiers:
            out.nSteps = np.maximum(out.nSteps,t.indexA)
        out.nSteps = out.nSteps + 1
        return out
    
    def assign(self,covs):
        '''Return a dictionary in which keys are tier numbers
        and values are vectors of indices of targets in that 
        tier.  Targets not in any tier are put in a list
        with tier number -1 as key.
        Input is assumed to be packed covariances.'''
        
        # Get compressed variables for each target and
        # convert them to bin numbers

        a = self.cc.bulkCompress(covs)

        # Assign grid indices to each target and form a 1-d index for them
        targetIndices = np.floor( (a[:,:self.dimA] -self.startA)/self.stepA).astype(int)
        ii = np.ravel_multi_index(targetIndices.T, self.nSteps,mode='clip')
        print(self.nSteps, np.min(targetIndices,axis=1), np.max(targetIndices,axis=1)) ###

        # Get the same things for the tiers
        tierIndices = np.vstack([t.indexA for t in self.tiers])
        jj = np.ravel_multi_index(tierIndices.T, self.nSteps,mode='clip')

        # Break up ii into sets with unique values
        isort = np.argsort(ii)
        tmp = ii[isort]
        # Get indices of those locations where a new tier index begins
        firsts = np.where(tmp[1:]!=tmp[:-1])[0] + 1
        firsts = np.append(firsts,tmp.shape[0])

        out = {}
        begin = 0
        for end in firsts:
            # See if each batch of targets is from a real tier.
            targetNum = tmp[begin]
            match = np.where(jj==targetNum)[0]
            if len(match)==0:
                # No matching tier
                print('No matching noise tier',
                          np.unravel_index(targetNum, self.nSteps),
                          'for {:d} sources'.format(end-begin))
                if -1 in out:
                    # Already have a no-tier list.  Add to it.
                    out[-1] = np.concatenate((out[-1], isort[begin:end]))
                else:
                    # Start a no-tier entry
                    out[-1] = np.array(isort[begin:end])
            elif len(match)==1:
                # Create entry for the matched tier
                tierNum = self.tiers[match[0]].id
                out[tierNum] = np.array(isort[begin:end])
                print('Tier {:d} gets {:d} targets'.format(tierNum, end-begin)) ###
            else:
                raise ValueError('Should not have gotten two matching tiers: '+str(match))

            begin = end
        return out

    def assignPQRSel(self, tab):
        '''Alters an Astropy Table of (pseudo-)targets to include columns:
        `noisetier:` assignmment to tier, -1 if none
        `select:`  integer value of 1 (0) if (not) passing selection
                criteria for its tier
        `pqr:` The selection PQR (per unit area) for the covariance
               of this target, if it does *not* pass selection criteria.

        Input table should have columns for `moments` and `covariance.`
        Output PQR is set to zero if the tier is -1 (does not fit anything).
        Galaxies that pass selection are left at incoming PQR value (if any),
          else set to 0.
        Returns the updated table'''
        

        # ??? Need to alter the below if we are using magnification or colors
        covs = None
        for k in 'COV_EVEN','cov_even','COVARIANCE','covariance',colnames['covariance']:
            if k in tab.colnames:
                covs = tab[k]
        if covs is None:
            raise ValueError('No covariance column in table')
        assignments = self.assign(covs)
        m = Moment()
        fluxes = None
        for k in 'moments','MOMENTS':
            if k in tab.colnames:
                fluxes = tab[k][:,m.M0]
        if fluxes is None:
            raise ValueError('No MOMENTS column in table')


        # Create or read columns for PQR, tier number, select.
        # Enforce uppercase
        if 'pqr' in tab.colnames:
            pqr = tab['pqr'].data.copy()
            del tab['pqr']
        elif colnames['pqr'] in tab.colnames:
            pqr = tab[colnames['pqr']].data.copy()
        else:
            pqr = np.zeros( (covs.shape[0],6), dtype=float)

        if 'select' in tab.colnames:
            select = tab['select'].data.copy()
            del tab['select']
        elif 'SELECT' in tab.colnames:
            select = tab['SELECT'].data.copy()
        else:
            select = np.zeros(covs.shape[0], dtype=bool)

        if 'noisetier' in tab.colnames:
            targetTier = tab['noisetier'].data.copy()
            del tab['noisetier']
        elif colnames['tierNumber'] in tab.colnames:
            targetTier = tab[colnames['tierNumber']].data.copy()
        else:
            targetTier = np.ones(covs.shape[0], dtype=int) * -1

        for t in self.tiers:
            if t.id not in assignments:
                # No targets for this tier
                continue
            use = assignments[t.id]
            targetTier[use] = t.id
            # Find objects failing selection
            inRange = np.ones_like(use, dtype=bool)
            if t.fluxMin is not None:
                inRange = np.logical_and(inRange, fluxes[use] >= t.fluxMin)
            if t.fluxMax is not None:
                inRange = np.logical_and(inRange, fluxes[use] < t.fluxMax)
            select[use[inRange]] = True
            use = use[~inRange]
            select[use] = False
            pqr[use,:] = t.pqrSel(covs[use])
        if -1 in assignments:
            # Make a PQR for certain deselection
            nullP = np.zeros_like(pqr[0])
            use = assignments[-1]
            pqr[use,:] = nullP
            select[use] = False

        # Update the table
        tab[colnames['pqr']] = pqr.astype(np.float32)
        tab['SELECT'] = select
        tab[colnames['tierNumber']] = targetTier.astype(np.int64)
        return tab

class PqrInterpolator:
    def __init__(self, nodeid, cov, pqr, compressor,
                     order=2):
        '''Create a function that returns a selection PQR (per unit area)
        given a covariance matrix for the (even) moments.  The inputs 
        are arrays with one row for each PQR calculated at a given set of
        moments.  The arrays are:
        `nodeid:`  An integer identifying locations in the compressed monopole space
        `cov:` Pack cov matrices
        `pqr:` Selection probabilities (and derivs) at those covariances.
        `compressor` is an instance of a CovarianceCompressor.  It is assumed
        that the last two elements are the quadrupole measures e1 and e2, while
        preceding elements are the space over which we'll interpolate.
        `order` is an integer, or a tuple matching the dimensionality of the interp
        space, that gives order of polynomial fit to the data at each dimension.
        '''

        '''In this code we transition between multiple spaces:  
        * The original packed covariance-element space
        * The compressed version of the covariance, the first
          self.NA of which are the "keys" for the interpolation,
          the next two of which are the e1/e2 for the quadrupole
          part of compressed covariance.
        * The "interpolation range" space, which is space of
          coefficients that can define a PQR
        * The PQR itself, which is a function of (e1,e2) and the
          range space.
        '''
        self.cc = compressor

        # Size of the PQR arrays
        self.P = 0
        self.Q1 = 1
        self.Q2 = 2
        self.R11 = 3
        self.R12 = 4
        self.R22 = 5
        self.NPQR = 6
        
        # Define standard indices in the range space to which we're interpolating
        # The map from range space to PQR is
        #    P = P0 + d2P * (e1^2 + e2^2)
        #    Q[1,2] = dQ * (e1,e2)
        #    R[11,22] = R0 + d2R * (e1^2 + e2^2)
        self.P0 = 0        
        self.R0 = 1
        self.dQ = 2
        self.d2P = 3
        self.d2R = 4
        self.NRANGE = 5
        
        # Get compress values for each node
        cdata = self.cc.bulkCompress(cov)
        
        # Number of interpolation keys - size of compressed vector minus 2
        self.NA = cdata.shape[1] - 2
        
        # Nominal values are those at first covariance
        self.nominalA = cdata[0, :self.NA]

        # Reduce each node's PQR values to P, R, dQ/de, d^2[P,R]/de^2
        # Find nodes that have more than one entry we can use
        nodes, count = np.unique(nodeid, return_counts=True)
        nodes = nodes[count>1]
        aa = []   # Interpolation keys for each node
        interp = []  # interpolation range values at each node
        # Get keys and range location for each node
        for n in nodes:
            use = nodeid==n
            aa.append(cdata[use,:self.NA][0] - self.nominalA)
            e1 = cdata[use,self.NA]
            e2 = cdata[use,self.NA+1]
            interp.append(self._fitPQRNode(pqr[use], e1, e2))

        # Now fit to variation of range with keys
        aa = np.array(aa)
        interp = np.array(interp)

        # Set up interpolator.
        if type(order) is int:
            self.order = (order,) * self.NA
        elif len(order) == self.NA:
            self.order = tuple(order)
        else:
            raise ValueError("Invalid interpolation order:",order)

        nTerms = np.prod(np.array(self.order)+1)
        if len(nodes) < nTerms:
            raise ValueError("Not enough covariance nodes to fit a PQR polynomial of desired order")
        
        self.coeffs = self._interpolateNodes(interp, aa, self.order)
        return

    def __call_OLD_BILINEAR__(self,covs):
        '''Return pqr's inferred for given (array of packed) covariances'''
        if covs.ndim==1:
            cdata = self.cc.compress(covs).reshape(1,-1)
        elif covs.ndim==2:
            cdata = self.cc.bulkCompress(covs)
        else:
            raise ValueError('covariance matrix input with invalid dimension')

        # Interpolate from keys to range space
        aa = cdata[:,:self.NA] - self.nominalA
        A = np.vstack( (np.ones_like(aa[:,0]),
                            aa[:,0],
                            aa[:,1],
                            aa[:,0]*aa[:,1])).T   # Ready for bilinear
        interp = np.dot(A, self.coeffs.T)
        # Now apply quadrupole corrections
        m = self._eMatrix(cdata[:,self.NA], cdata[:,self.NA+1])
        pqr = np.einsum('ik,ijk->ij',interp, m)
        return pqr
    
    def __call__(self,covs):
        '''Return pqr's inferred for given (array of packed) covariances'''
        if covs.ndim==1:
            cdata = self.cc.compress(covs).reshape(1,-1)
        elif covs.ndim==2:
            cdata = self.cc.bulkCompress(covs)
        else:
            raise ValueError('covariance matrix input with invalid dimension')

        # Evaluate the compressed pqr at each covariance, one component at a time.
        aa = cdata[:,:self.NA] - self.nominalA
        interp = np.vstack( [poly(aa) for poly in self.coeffs]).T

        # Now calculate needed powers in ellipticity
        m = self._eMatrix(cdata[:,self.NA], cdata[:,self.NA+1])
        pqr = np.einsum('ik,ijk->ij',interp, m)
        return pqr

    def _eMatrix(self,e1,e2):
        '''Produce the matrix embodying the ellipticity behavior, namely
        Standard order of these coefficients is P0, R0, dQ, d2P, d2R
        '''
        npts = e1.shape[0]
        m = np.zeros((npts, self.NPQR, self.NRANGE), dtype=float)
        m[:,self.P, self.P0] = 1.
        m[:,self.R11, self.R0] = 1.
        m[:,self.R22, self.R0] = 1.
        esq = e1*e1 + e2*e2
        m[:,self.P, self.d2P] = esq
        m[:,self.R11, self.d2R] = esq
        m[:,self.R22, self.d2R] = esq
        m[:,self.Q1, self.dQ] = e1
        m[:,self.Q2, self.dQ] = e2
        return m

    def _fitPQRNode(self,pqr,e1,e2):
        '''Find the e coefficients that fit PQRs measured at one node'''
        m = self._eMatrix(e1,e2)
        return np.linalg.lstsq(m.reshape([-1,self.NRANGE]), pqr.flatten(), rcond=None)[0]

    def _interpolateNodes_OLD_BILINEAR(self,interp,aa):
        '''Produce the matrix of (NRANGE x 4) coeffs that
        models bilinear variation of the interpolator range on its keys.
        `interp` is a NNODESxNRANGE matrix of data being fit
        `aa` is NNODESx2 array of interpolation keys'''
        # There is essentially a separate linear solution for each
        # element of the interpolation range
        A = np.vstack( (np.ones_like(aa[:,0]),
                            aa[:,0],
                            aa[:,1],
                            aa[:,0]*aa[:,1])).T   # Ready for bilinear
        coeffs = np.linalg.lstsq(A, interp, rcond=None)[0]

        return coeffs.T

    def _interpolateNodes(self,interp,aa,order):
        '''Produce a set of polynomials to interpolate each element of aa.'''

        # Make a separate polynomial for each element of the interpolation space.
        # Note this is non-ideal in that we recalculate polyomial terms for each column.
        return [PolyND.fit(aa, y, order) for y in interp.T]


class PolyND:
    def __init__(self, coeffs):
        '''Class holding an arbitrary n-dimensional polynomial expression.
        Can be initialized with a coefficient array (whose shape defines the
        order in each variable).'''
        self.coeffs = np.array(coeffs)
        self.order = np.array(self.coeffs.shape) - 1
        return

    @classmethod
    def _buildTerms(cls,x,order):
        '''Return an array of all terms x[0]^j0 * x[1]^j1 *.... x[n]^jn where the
        powers ji are <= order[i].  The last dimension
        of x runs over the dimensionality of the polynomial.  Additional 
        leading dimensions are preserved.  The output will replace the lasst dimension so it runs over
        all of the combinations of orders.'''
        if x.shape[-1] != len(order):
            raise ValueError('Coefficient array is not of same size as order array')

        otherDims = tuple(x.shape[:-1])
        
        # Start with a unity value
        out = np.ones( x.shape[:-1]+(1,), dtype=float)
        nTerms = 1
        for idim in range(len(order)):
            # Each iteration of this loop will expand the dim of the last column
            # by a factor of the order of its polynomial
            powers = np.power(x[...,idim,np.newaxis],np.arange(order[idim]+1))
            out = out[...,np.newaxis] * powers[...,np.newaxis,:]
            nTerms *= order[idim]+1
            out = out.reshape(otherDims + (nTerms,))

        return out

    def __call__(self,x):
        '''Evaluate polynomial at position(s) x.  The last dimension of x
        should be the dimensionality of polynomial arguments.  Other dimensions
        describe array of evaluation points.  Output has shape of these other dims.
        '''
        return np.dot(self._buildTerms(x, self.order), self.coeffs.flatten())

    @classmethod
    def fit(cls, x, y, order):
        '''Return an instance of this class which has coefficients chosen to yield the
        minimal RMS difference from y when evaluated on x.  x should have one more dimension than
        y, with this last dimension spanning the space of polynomial arguments.
        '''
        # We'll do a least-squares minimization to y = A * coeff, where A are the terms
        # of polynomial.
        oo = np.array(order)
        A = cls._buildTerms(x, oo)
        # Flatten all dimensions of sampling
        A = A.reshape(-1, A.shape[-1])
        ### Do multiple y's at once...
        yy = y.flatten()
        coeffs = np.linalg.lstsq(A,yy, rcond=None)[0]
        coeffs = coeffs.reshape( *(oo+1))
        return PolyND(coeffs)


