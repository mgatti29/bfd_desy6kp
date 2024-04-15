# Manipulations of PQR's 

from .keywords import colnames
import numpy as np

# Note that conventional order for PQR is defined in BfdConfig.h has
# [p, dp/dg1, dp/dg2, d^2p/dg1^2, d^2p/dg1 dg2, d^2p/dg2^2]

# If we have magnification, there are 10 elements:
# [p,
#  dp/dg1, dp/dg2, dp/dmu,
#  d^2p/dg1^2, d^2p/dg1 dg2, d^2p/dg2^2
#  d^2p/dg1 dmu, d^2p/dg2 dmu, d^2p/dmu^2]

# Map from numpy types to complex type of same precision
complexTypeFor = {np.float32:           np.complex64,
                  np.dtype('float32'):  np.complex64,
                  np.dtype('complex64'):np.complex64,
                  np.dtype('>f4'):      np.complex64,
                  np.float64:           np.complex128,
                  np.dtype('float64'):  np.complex128,
                  np.dtype('complex128'):np.complex128,
                  np.dtype('>f8'):      np.complex128,
                  np.float128:          np.complex256,
                  np.dtype('float128'): np.complex256,
                  np.dtype('complex256'):np.complex256}

# Map from numpy types to real type of same precision
realTypeFor =    {np.float32:           np.float32,
                  np.dtype('float32'):  np.float32,
                  np.dtype('complex64'):np.float32,
                  np.dtype('>f4'):      np.float32,
                  np.float64:           np.float64,
                  np.dtype('float64'):  np.float64,
                  np.dtype('complex128'):np.float64,
                  np.dtype('>f8'):      np.float64,
                  np.float128:          np.float128,
                  np.dtype('float128'): np.float128,
                  np.dtype('complex256'):np.float128}
    
class Pqr:
    '''This class just has constants defining array positions for various
    derivatives of some quantity w.r.t. shear + magnification.
    Transformations will not be made part of this class since the
    instances of PQR's will generally just be arrays, so we will make
    them functions instead of methods.'''

    HAS_MU = True
    
    D0 = 0
    DG1 = 1
    DG2 = 2
    DMU = 3
    DG1DG1 = 4
    DG1DG2 = 5
    DG2DG2 = 6
    DG1DMU = 7
    DG2DMU = 8
    DMUDMU = 9
    NPQR = 10
    QSlice = slice(DG1,DG1DG1)
    RSlice = slice(DG1DG1,NPQR)
    RShape = (3,3)

    # Entries to take from linear version to make matrix R:
    BUILDR = ([DG1DG1,DG1DG2,DG1DMU,DG1DG2,DG2DG2,DG2DMU,DG1DMU,DG2DMU,DMUDMU],)
    # Entries to take from R to build linear
    GETR = ([0, 0, 1, 0, 1, 2], [0, 1, 1, 2, 2, 2])
    

    # Indices for the complex form of derivatives (D0 is same)
    DU = 1  # 1st deriv wrt mag
    DV = 2  # 1st deriv wrt complex shear
    DVb= 3  # 1st deriv wrt complex shear conjugate
    DUDU = 4  # 2nd deriv wrt magnification
    DUDV = 5  # 2nd deriv wrt mag & complex shear
    DUDVb= 6  # 2nd deriv wrt mag & complex shear conjugate
    DVDV = 7  # 2nd deriv wrt complex shear
    DVbDVb=8  # 2nd deriv wrt complex shear conjugate
    DVDVb= 9  # 2nd deriv wrt complex shear & conjugate
    NCD = 10  # Number of derivatives


class PqrNoMu:
    '''This class just has constants defining array positions for various
    derivatives of some quantity w.r.t. shear WITHOUT MAGNIFICATION
    Transformations will not be made part of this class since the
    instances of PQR's will generally just be arrays, so we will make
    them functions instead of methods.'''

    HAS_MU = False

    D0 = 0
    DG1 = 1
    DG2 = 2
    DG1DG1 = 3
    DG1DG2 = 4
    DG2DG2 = 5
    NPQR = 6
    QSlice = slice(DG1,DG1DG1)
    RSlice = slice(DG1DG1,NPQR)
    RShape = (2,2)
    # Entries to take from linear version to make matrix R:
    BUILDR = ([DG1DG1,DG1DG2,DG1DG2,DG2DG2],)
    # Entries to take from R to build linear
    GETR = ([0, 0, 1], [0, 1, 1])

    # Indices for the complex form of derivatives (D0 is same)
    DV = 1
    DVb = 2
    DVDV = 3
    DVbDVb = 4
    DVDVb = 5
    NCD = 6
    
    
def pickPqr(a, axis=-1):
    '''Return the indexing class (with or without mu) based
    on the size of the relevant axis of array'''
    if a.shape[axis]==Pqr.NPQR:
        return Pqr
    elif a.shape[axis]==PqrNoMu.NPQR:
        return PqrNoMu
    else:
        raise ValueError('PQR axis of invalid length = ' + str(a.shape[axis]))

def pqrCofactor(lens, axis=-1):
    '''From an input array that has some axis indexing (g1, g2, mu) or (g1,g2)
    construct an array that has indices up to `NPQR` along that axis which can be
    dotted into a `Pqr` (or `PqrNoMu`) vector to yield the value of the Taylor expansion
    represented by the `Pqr` evaluated at the input vector.  Size of the chosen
    axis will determine whether mu terms are produced.'''

    n = lens.shape[axis]
    if n==2:
        p = PqrNoMu
    elif n==3:
        p = Pqr
    else:
        raise ValueError('lensing array must have 2 or 3 components, has ' + str(lens.shape[axis]))

    tmp = np.swapaxes(lens, axis, -1)
    out = np.ones( tmp.shape[:-1] + (p.NPQR,), dtype=lens.dtype)
    out[...,p.DG1] = lens[...,0]
    out[...,p.DG2] = lens[...,1]
    out[...,p.DG1DG1] = 0.5*lens[...,0]**2
    out[...,p.DG2DG2] = 0.5*lens[...,1]**2
    out[...,p.DG1DG2] = lens[...,1] * lens[...,0]
    if n==3:
        # Add the mu terms
        out[...,p.DMU] = lens[...,2]
        out[...,p.DMUDMU] = 0.5*lens[...,2]**2
        out[...,p.DG1DMU] = lens[...,2] * lens[...,0]
        out[...,p.DG2DMU] = lens[...,2] * lens[...,1]
    return np.swapaxes(out,-1,axis)
        
def stripMuPqr(pqr, axis=-1):
    '''Return a new version of input PQR that has all mu derivs (DMU) stripped.
    Input:
    axis  = which axis is indexing the derivatives'''
    s = [slice(None)] * pqr.ndim
    use = [0]*PqrNoMu.NPQR
    use[PqrNoMu.D0] = Pqr.D0
    use[PqrNoMu.DG1] = Pqr.DG1
    use[PqrNoMu.DG2] = Pqr.DG2
    use[PqrNoMu.DG1DG1] = Pqr.DG1DG1
    use[PqrNoMu.DG1DG2] = Pqr.DG1DG2
    use[PqrNoMu.DG2DG2] = Pqr.DG2DG2
    s[axis] = use
    return np.array(pqr[tuple(s)])

def stripMuUvv(uvv, axis=-1):
    '''Return a new version of input UVVb that has all mu derivs (DU) stripped.
    Input:
    axis  = which axis is indexing the derivatives'''
    s = [slice(None)] * uvv.ndim
    use = [0]*PqrNoMu.NCD
    use[PqrNoMu.D0] = Pqr.D0
    use[PqrNoMu.DV] = Pqr.DV
    use[PqrNoMu.DVb] = Pqr.DVb
    use[PqrNoMu.DVDV] = Pqr.DVDV
    use[PqrNoMu.DVbDVb] = Pqr.DVbDVb
    use[PqrNoMu.DVDVb] = Pqr.DVDVb
    s[axis] = use
    return np.array(uvv[tuple(s)])


''' Desired functions:
Rotation
yflip?
'''
def splitPqr(pqr, axis=-1):
    '''Break apart a PQR into a scalar p, first derivs q, and 2nd deriv matrix r.
    Size of input axis determines whether mu derivatives are present.
    Outputs are new arrays, not views

    Inputs:
    pqr  = array of values/derivatives in packed PQR form
    axis = which axis indexes the derivatives.

    Outputs:
    p    = P scalars, in array 1 dimension lower than input
    q    = Q vectors, same input dimension but derivatives in LAST INDEX
    r    = R matrices, 1 dimension larger than last, matrix in LAST 2 INDICES
    '''
    
    pp = pickPqr(pqr, axis=axis)

    # Place pqr axis last - probably a view
    src = np.moveaxis(pqr, axis, -1)
    
    p = np.array(src[...,pp.D0])
    q = np.array(src[...,pp.QSlice])
    s = src.shape[:-1] + pp.RShape
    r = np.array(src[...,pp.BUILDR].reshape(s))
    
    return p,q,r

def packPqr(p,q,r,axis=-1):
    '''Pack scalar p, vector q, (symmetric) matrix r into a linear PQR axis.
    Size of q determines whether mu derivatives are present. Data type is
    preserved.
    Inputs:
    p, q, and r: arrays of size [...],[...,N],[...,N,N] where N is 2 or 3,
          holding value, first, and second derivs, respectively.
    axis  = which axis in the output array to have as pqr index

    Outputs:
    p    = P scalars, in array 1 dimension lower than input
    q    = Q vectors, same input dimension but derivatives in LAST INDEX
    r    = R matrices, 1 dimension larger than last, matrix in LAST 2 INDICES
    '''
    if q.shape[-1]==3:
        pp = Pqr
    elif q.shape[-1]==2:
        pp = PqrNoMu
    else:
        raise ValueError('Bad axis length = {:d} for Q vector'.format(q.shape[-1]))
    
    # Construct output array
    ss = q.shape[:-1]
    pqr = np.zeros(ss + (pp.NPQR,), dtype=q.dtype)
    # Fill it
    pqr[...,pp.D0] = p
    pqr[...,pp.QSlice] = q
    pqr[...,pp.RSlice] = r[ (slice(None),) * (q.ndim-1) + pp.GETR]

    # Move axis as desired
    pqr = np.moveaxis(pqr,-1,axis)
    return pqr

def pqr2Uvv(pqr, axis=-1):
    '''Convert derivatives w.r.t. g1,g2[,mu] to derivatives
    w.r.t. V=(g1+j*g2), Vb=(g1-j*g2)[, U=mu].  Output is complex.
    Input:
    axis  = which axis in the output array to have as pqr index
    Output:
    Complex array with multipole representation in the chosen axis.'''
    pp = pickPqr(pqr, axis=axis)

    # Create destination array
    s = list(pqr.shape)
    s[axis] = pp.NCD
    otype = complexTypeFor[pqr.dtype]
    uvv = np.zeros(s, dtype=otype)

    # Put axis of interest last
    uvv = np.moveaxis(uvv, axis, -1)
    src = np.moveaxis(pqr, axis, -1)

    j32 = np.complex64(1j)
    
    uvv[...,pp.D0] = src[...,pp.D0]
    uvv[...,pp.DV] = 0.5*(src[...,pp.DG1] - j32*src[...,pp.DG2])
    uvv[...,pp.DVb] = 0.5*(src[...,pp.DG1] + j32*src[...,pp.DG2])
    uvv[...,pp.DVDV] = 0.25*(src[...,pp.DG1DG1] -(2*j32) * src[...,pp.DG1DG2] - src[...,pp.DG2DG2])
    uvv[...,pp.DVDVb] = 0.25*(src[...,pp.DG1DG1] + src[...,pp.DG2DG2])
    uvv[...,pp.DVbDVb] = 0.25*(src[...,pp.DG1DG1] +(2*j32) * src[...,pp.DG1DG2] - src[...,pp.DG2DG2])
    if pp.HAS_MU:
        uvv[...,pp.DU] = src[...,pp.DMU]
        uvv[...,pp.DUDV] = 0.5*(src[...,pp.DG1DMU] - j32*src[...,pp.DG2DMU])
        uvv[...,pp.DUDVb] = 0.5*(src[...,pp.DG1DMU] + j32*src[...,pp.DG2DMU])
        uvv[...,pp.DUDU] = src[...,pp.DMUDMU]

    # Put axis back
    uvv = np.moveaxis(uvv, -1, axis)
    return uvv

def uvv2Pqr(uvv, axis=-1):
    '''Create derivatives w.r.t. g1,g2[,mu] from derivatives
    w.r.t. V=(g1+j*g2), Vb=(g1-j*g2)[, U=mu].  Output is complex.
    Input:
    axis  = which axis in the output array to have as uvv index
    Output:
    Complex array with pqr representation in the chosen axis.'''
    pp = pickPqr(uvv, axis=axis)

    # Create destination array
    s = list(uvv.shape)
    s[axis] = pp.NPQR
    pqr = np.zeros(s, dtype=uvv.dtype)

    # Put axis of interest last
    pqr = np.moveaxis(pqr, axis, -1)
    src = np.moveaxis(uvv, axis, -1)

    j32 = np.complex64(1j)
    
    pqr[...,pp.D0] = src[...,pp.D0]

    pqr[...,pp.DG1] = src[...,pp.DV] + src[...,pp.DVb]
    pqr[...,pp.DG2] = j32 * (src[...,pp.DV] - src[...,pp.DVb])

    pqr[...,pp.DG1DG1] = src[...,pp.DVDV] + 2*src[...,pp.DVDVb] + src[...,pp.DVbDVb]
    pqr[...,pp.DG1DG2] = j32*(src[...,pp.DVDV] - src[...,pp.DVbDVb])
    pqr[...,pp.DG2DG2] = -src[...,pp.DVDV] + 2*src[...,pp.DVDVb] - src[...,pp.DVbDVb]
    
    if pp.HAS_MU:
        pqr[...,pp.DMU] = src[...,pp.DU]
        pqr[...,pp.DG1DMU] = src[...,pp.DUDV] + src[...,pp.DUDVb]
        pqr[...,pp.DG2DMU] = j32 * (src[...,pp.DUDV] - src[...,pp.DUDVb])
        pqr[...,pp.DMUDMU] = src[...,pp.DUDU]

    # Move axis back
    pqr = np.moveaxis(pqr, -1, axis)
    return pqr

def logPqr(pqr, axis=-1):
    '''Convert an array representing quadratic Taylor 
    expansion of p w.r.t. 2 or 3 quantities into Taylor expansion of log(P).
    Any input rows with p<=0 are put as zeros in output.
    Values of input and output are packed in last dimension of an array.
    Other dimensions are preserved.'''
    p,q,r = splitPqr(pqr, axis=axis)
    good = p>0.
    q = q / p[...,np.newaxis]
    r = r / p[...,np.newaxis, np.newaxis]
    # Subtract outer product of 1st derivs
    r = r - q[...,:,np.newaxis] * q[...,np.newaxis,:]
    p = np.log(p)
    out = packPqr(p,q,r, axis=-1)
    out = np.where(good[...,np.newaxis],out,0.)

    # Move pqr axis back to original location
    out = np.moveaxis(out, -1, axis)
    return out

def stampizePqr(pqr, axis=-1):
    '''Adjust PQR for powers of (1+mu) from Poisson mode
    to the power appropriate to postage stamp mode.'''
    
    pp = pickPqr(pqr, axis=axis)
    if not pp.HAS_MU:
        # Nothing needs to be done
        return pqr

    out = pqr.copy()
    # Move pqr axis to the back
    out = np.moveaxis(out, axis, -1)
    pqr2 = np.moveaxis(pqr, axis, -1)

    out[...,[pp.DMU,pp.DG1DMU,pp.DG2DMU]] += 2*pqr2[...,[pp.D0,pp.DG1,pp.DG2]]
    out[...,pp.DMUDMU] += 4*pqr[...,pp.DMU] + 2*pqr2[...,pp.D0]
    
    # Move axis back
    out = np.moveaxis(out, -1, axis)
    return out

def oneMinusPqr(pqr, axis=-1):
    '''Convert Taylor expansion of p into expansion of 1-p
       axis = which axis indexes the pqr'''
    out = -1. * pqr
    s = [slice(None)] * out.ndim
    s[axis] = Pqr.D0
    out[tuple(s)] += 1.
    return out

def rotatePqr(theta, pqr, axis=-1):
    '''Return pqr that expresses the same derivatives with the AXES defining
    g1 and g2 rotated CLOCKWISE by theta. Data type of pqr is preserved.
    Parameters:
    `theta`: array of rotation angles that will be broadcast into the non-pqr axes.
    `pqr`: array with one axis indexing the derivatives 
           with respect to g1 and g2 (and mu) in PQR order
    `axis` : which axis indexes the pqr derivatives.
    Returns:
    Rotated version of `pqr` with same shape as original. '''

    pp = pickPqr(pqr, axis=axis)
    
    c2 = np.cos(2*theta)
    s2 = np.sin(2*theta)

    out = pqr.copy()
    # Move pqr axis to the back
    out = np.moveaxis(out, axis, -1)

    # Rotate 1st derivs
    tmp = out[...,pp.DG1]*c2 - out[...,pp.DG2]*s2
    out[...,pp.DG2] = out[...,pp.DG2]*c2 + out[...,pp.DG1]*s2
    out[...,pp.DG1] = tmp

    if pp.HAS_MU:
        # Rotate DUDG derivs
        tmp = out[...,pp.DG1DMU]*c2 - out[...,pp.DG2DMU]*s2
        out[...,pp.DG2DMU] = out[...,pp.DG2DMU]*c2 + out[...,pp.DG1DMU]*s2
        out[...,pp.DG1DMU] = tmp

    # Rotate 2nd derivs
    c4 = c2*c2-s2*s2
    s4 = 2*c2*s2
    e0 = 0.5 * (out[...,pp.DG1DG1] + out[...,pp.DG2DG2])
    e1 = 0.5 * (out[...,pp.DG1DG1] - out[...,pp.DG2DG2])
    e2 = out[...,pp.DG1DG2]
    e1, e2 = e1*c4-e2*s4, e1*s4+e2*c4
    out[...,pp.DG1DG1] = e0 + e1
    out[...,pp.DG1DG2] = e2
    out[...,pp.DG2DG2] = e0 - e1

    # Move axis back
    out = np.moveaxis(out, -1, axis)
    return out

def rotateUvv(theta, uvv, axis=-1):
    '''Return UVV that expresses the same derivatives  with the AXES defining
    g1 and g2 rotated CLOCKWISE by theta. Data type of pqr is preserved.
    Parameters:
    `theta`: array of rotation angles that will be broadcast into the non-pqr axes.
    `uvv`: complex array with one axis indexing the derivatives 
           with respect to g1 and g2 (and mu) in UVV order
    `axis` : which axis indexes the UVV derivatives.
    Returns:
    Rotated version of `uvv` with same shape as original. '''

    # Get type and format of the data
    CT = complexTypeFor[uvv.dtype] # Our complex type
    J = CT(1j)   # Imaginary unit in our type
    if uvv.shape[axis]==Pqr.NCD:
        pp = Pqr
    elif uvv.shape[axis]==PqrNoMu.NCD:
        pp = PqrNoMu
    else:
        raise ValueError('Bad UVV axis length ' + str(uvv.shape[axis]))

    z2 = np.exp(-2*J*theta, dtype=CT)
    out = uvv.copy()
    # Move pqr axis to the back
    out = np.moveaxis(out, axis, -1)

    # Rotate 1st derivs
    out[...,pp.DV] *= z2
    out[...,pp.DVb] *= np.conj(z2)
    if pp.HAS_MU:
        out[...,pp.DUDV] *= z2
        out[...,pp.DUDVb] *= np.conj(z2)
    z4 = z2 * z2
    out[...,pp.DVDV] *= z4
    out[...,pp.DVbDVb] *= np.conj(z4)

    # Move axis back
    out = np.moveaxis(out, -1, axis)
    return out

def yflipPqr(pqr, axis=-1):
    '''Return pqr that expresses the same derivatives  with the axes defining
    g1 and g2 rotated flipped about x axis, y->-y. Basically just negating derivative
    with respect to g2. Data type of pqr is preserved.
    Parameters:
    `pqr`: array with one axis indexing the derivatives 
           with respect to g1 and g2 (and mu) in PQR order
    `axis` : which axis indexes the pqr derivatives.
    Returns:
    Flipped version of `pqr` with same shape as original. '''
    pp = pickPqr(pqr, axis=axis)
    flipMe = [pp.DG2, pp.DG1DG2]
    if pp.HAS_MU:
        flipMe.append(pp.DG2DMU)
    out = pqr.copy()
    s = [slice(None)] * pqr.ndim
    s[axis] = flipMe
    out[tuple(s)] *= -1.
    return out

def maxPqr(pqr):
    '''Determine the value of x for which f= p + q*x + 1/2 x^T * r * x 
    is maximized.  If f is log(likelihood) then this is the max likelihood
    point.
    Returns the maximized value p(x_max), x_max, and (-r)^{-1}, which will be
    the covariance matrix if f = log(prob).
    This returns the shear (and maybe magnification) estimate.  The PQR is
    to be in the last axis of the input array, and the returned results
    have dimensions to match the input.'''

    p,q,r = splitPqr(pqr)
    cov = np.linalg.inv(-r)
    xmax = np.einsum('...ij,...j->...i',cov,q)
    val = p - 0.5*np.sum(xmax*q, axis=-1)
    return val, xmax, cov

def meanShear(pqr):
    '''Extract most likely shear and covariance matrix from PQR
    representing log(p).'''
    return maxPqr(pqr)[1:]

def sumPqr(tab):
    '''Extract the total PQR for the log of probability vs shear
    given an input `TargetTable` with per-detection information. The
    treatment of non-detections can use two possible formulae: 
    when `stampMode=True` is in the table's metadata, we assume that there is a fixed number
    of target placements, and every row reports either a selected
    galaxy or a non-selected one, the latter including non-detections.
    For `stampMode=False` (or "Poisson" mode), we assume that galaxies
    have been placed by a Poisson process, and that there are 
    rows representing "pseudo-detections" - i.e. reporting the search
    area at a given level of noise.

    The table should have these columns:
    `select`: A column which is non-zero for targets that have been selected,
       and zero for unselected detections and pseudo-detections.
    `area`: In stamp mode, this area gives the number of galaxy placements
       represented by the row (which can potentially be >1 if reporting
       non-detected placements.  If the entry is zero or the column is absent,
       a value of 1. will be assumed.

       In Poisson mode, this column should be zero for galaxy detections (whether
       selected or not) and should give the sky area for pseudo-detections.
    `pqr`: Taylor expansion wrt shear of probability (*not* log).
       In stamp mode, this is equal to the *detection* probability per stamp for
       any non-selected row.  In Poisson mode, this is equal to the *detection* 
       probability per unit area for any pseudo-detection.  Targets with p=0 are
       ignored.

    The output is the total log(P), and its derivs w.r.t. shear, for the full
    ensemble.'''

    if 'area' in tab.colnames:
        area = tab['area']
        if tab.stampMode:
            # Every entry is one stamp (at least)
            area = np.maximum(1., area)
    elif colnames['area'] in tab.colnames:
        # Using an old name for the column
        area = tab[colnames['area']]
        if tab.stampMode:
            # Every entry is one stamp (at least)
            area = np.maximum(1., area)
    else:
        if tab.stampMode:
            # Every entry is one stamp
            area = np.ones(len(tab), dtype=float)
        else:
            raise ValueError('sumPqr requires area column in Poisson mode')


    if 'select' in tab.colnames:
        select = tab['select'] > 0
    else:
        select = tab[colnames['select']]>0

    if 'pqr' in tab.colnames:
        pqr = tab['pqr']
    else:
        pqr = tab[colnames['pqr']]
    use = pqr[:,0] > 0.

    # First sum the log(PQR)'s for selected targets
    # ALL SUMMATIONS EXPLICITLY USE 64 BITS!!
    ss = np.logical_and(use, select)

    if tab.stampMode:
        # Change the mu derivs to remove area factor
        out = np.sum( logPqr(stampizePqr(pqr[ss])), axis=0, dtype=float)
        # Sum up log(1-p) for all deselected and pseudo detections
        ss = np.logical_and(use, ~select)
        # Don't use non-detections with 100% detection probability
        ss = np.logical_and(ss, pqr[:,0]<0.999)
        # Adjust any mu derivs for stamp mode
        nondet = oneMinusPqr(stampizePqr(pqr[ss]))

        out += np.sum( logPqr(nondet) * area[ss, np.newaxis], axis=0, dtype=float)
    else:
        out = np.sum( logPqr(pqr[ss]), axis=0, dtype=float)
        # Poisson mode: each pseudo-detection contributes
        # log (exp(- p * A)) where p is prob per unit area
        ss = np.logical_and(use, area>0)
        # The pseudo-detections should be non-selected as well, 
        # so throw that in for good measure
        ss = np.logical_and(ss, ~select)
        out -= np.sum(pqr[ss] * area[ss,np.newaxis], axis=0,  dtype=float)

    return out

def predictSelect(targets, tierColl, stampMode=False, shear=(0.,0.)):
    '''Calculate the observed and predicted rate of target selection.
    In stamp mode, this is given as selections per stamp.  In Poisson
    mode, it is selections per unit area.

    `targets` is a table of target info
    `tierColl` is a TierCollection object
    `shear` is the shear applied to the field

    Returns:  observed, predicted, and expected std. dev. of selection rate.'''
    
    if colnames['area'] in targets.columns.names:
        area = targets[colnames['area']]
        if stampMode:
            # Every entry is one stamp (at least)
            area = np.maximum(1., area)
    else:
        if stampMode:
            # Every entry is one stamp
            area = np.ones(len(targets), dtype=float)
        else:
            raise ValueError('sumPqr requires AREA column in Poisson mode')

    pqr = targets[colnames['pqr']]
    # eliminate some possible forms of junk, objects outside of usable areas:
    use = np.logical_and(pqr[:,0] > 0., targets[colnames['tierNumber']]>=0)
    select = np.logical_and(targets[colnames['select']], use)
    deselect = np.logical_and(~targets[colnames['select']], use)

    # Total survey area and selection counts:
    nSelect = np.count_nonzero(select)
    if stampMode:
        totalArea = np.sum(area[use], dtype=float)
    else:
        totalArea = np.sum(area[deselect], dtype=float)
    observedRate = nSelect / totalArea

    # Now accumulate the expected yield over all of the area
    if stampMode:
        # Sums are over all stamps, selected or not
        footprint = np.array(use)
    else:
        # Sums are only over the pseudo-detections
        footprint = np.logical_and(deselect, area>0.)
    
    tab = targets[footprint]
    tset = np.unique(tab[colnames['tierNumber']])
    sumP = 0.
    sumVar = 0.
    gTerms = np.array([1., shear[0], shear[1],
                       0.5*shear[0]*shear[0], shear[0]*shear[1],0.5*shear[1]*shear[1]])
    for t in tset:
        use = tab[colnames['tierNumber']]==t
        pqr = tierColl.tiers[t].pqrSel(tab[colnames['covariance']][use])
        p = np.dot(pqr, gTerms)
        a = area[footprint][use]
        sumP += np.dot(p,a)
        if stampMode:
            # Binomial variance
            sumVar += np.dot(p*(1-p), a)
        else:
            # Poisson variance
            sumVar += np.dot(p,a)

    return observedRate, sumP / totalArea, np.sqrt(sumVar) / totalArea
        