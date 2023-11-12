# Manipulations of PQR's for shear-only situations

import numpy as np
from .keywords import *
from scipy.special import erf

# Note that conventional order for PQR is defined in BfdConfig.h has
# [p, dp/dg1, dp/dg2, d^2p/dg1^2, d^2p/dg1 dg2, d^2p/dg2^2]

def logPQR(pqr):
    '''Convert an Nx6 array representing quadratic Taylor 
    expansion of p w.r.t. shear into Taylor expansion of log(P).
    Any input rows with p<=0 are put as zeros in output'''
    out = np.zeros_like(pqr)
    p = pqr[:,0]
    good = p>0.
    q = pqr[:,1:3] / p[:,np.newaxis]
    r = pqr[:,3:] / p[:,np.newaxis]
    out[:,0] = np.log(p)
    out[:,1:3] = q
    r[:,0] -= q[:,0]*q[:,0]
    r[:,1] -= q[:,1]*q[:,0]
    r[:,2] -= q[:,1]*q[:,1]
    out[:,3:] = r
    out = np.where(good[:,np.newaxis],out,0.)
    return out

def oneMinusPQR(pqr):
    '''Convert Taylor expansion of p into expansion of 1-p'''
    out = -1. * pqr
    out[:,0] += 1.
    return out

def meanShear(pqr):
    '''Extract most likely shear and covariance matrix from PQR
    representing log(p).'''
    q = pqr[1:3]
    r = np.zeros((2,2), dtype=float)
    r[0,0] = pqr[3]
    r[1,0] = r[0,1] = pqr[4]
    r[1,1] = pqr[5]
    cov = np.linalg.inv(-r)
    g = -np.dot(cov,q)
    return g,cov

def sumPQR(tab, stampMode = False):
    '''Extract the total PQR for the log of probability vs shear
    given an input table with per-detection information. The
    treatment of non-detections can use two possible formulae: 
    when `stampMode=True`, we assume that there is a fixed number
    of target placements, and every row reports either a selected
    galaxy or a non-selected one, the latter including non-detections.
    For `stampMode=False` (or "Poisson" mode), we assume that galaxies
    have been placed by a Poisson process, and that there are 
    rows representing "pseudo-detections" - i.e. reporting the search
    area at a given level of noise.

    The table should have these columns:
    `SELECT`: A column which is non-zero for targets that have been selected,
       and zero for unselected detections and pseudo-detections.
    `AREA`: In stamp mode, this area gives the number of galaxy placements
       represented by the row (which can potentially be >1 if reporting
       non-detected placements.  If the entry is zero or the column is absent,
       a value of 1. will be assumed.

       In Poisson mode, this column should be zero for galaxy detections (whether
       selected or not) and should give the sky area for pseudo-detections.
    `PQR`: Taylor expansion wrt shear of probability (*not* log).
       In stamp mode, this is equal to the *detection* probability per stamp for
       any non-selected row.  In Poisson mode, this is equal to the *detection* 
       probability per unit area for any pseudo-detection.  Targets with p=0 are
       ignored.

    The output is the total log(P), and its derivs w.r.t. shear, for the full
    ensemble.'''

    if colnames['area'] in tab.columns.names:
        area = tab[colnames['area']]
        if stampMode:
            # Every entry is one stamp (at least)
            area = np.maximum(1., area)
    else:
        if stampMode:
            # Every entry is one stamp
            area = np.ones(len(tab), dtype=float)
        else:
            raise ValueError('sumPQR requires AREA column in Poisson mode')

    select = tab[colnames['select']]>0

    pqr = tab[colnames['pqr']]
    use = pqr[:,0] > 0.

    # First sum the log(PQR)'s for selected targets
    # ALL SUMMATIONS EXPLICITLY USE 64 BITS!!
    ss = np.logical_and(use, select)
    out = np.sum( logPQR(pqr[ss]), axis=0, dtype=float)

    if stampMode:
        # Sum up log(1-p) for all deselected and pseudo detections
        ss = np.logical_and(use, ~select)
        # Don't use non-detections with 100% detection probability
        ss = np.logical_and(ss, pqr[:,0]<0.999)
        nondet = oneMinusPQR(pqr[ss])

        out += np.sum( logPQR(nondet) * area[ss, np.newaxis], axis=0, dtype=float)
    else:
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
            raise ValueError('sumPQR requires AREA column in Poisson mode')

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
        
    
#### Code below will calculate the selection probability (but no shear derivs yet).
# Make a function for integral of Gaussian above lower limit
def eFunc(x):
    '''Return integral of Gaussian above value x.
    Returns integral, and first and second derivs w.r.t. x'''
    ss = np.sign(x)
    x = x*ss
    e = 0.5*erf(x/np.sqrt(2.))
    de = np.exp(-0.5*x*x) / np.sqrt(2*np.pi)
    d2e = -x * de
    return 0.5-e*ss, -de, -d2e*ss

def eFunc2(x):
    '''Return integral of Gaussian above value x.
    Returns integral, and first and second derivs w.r.t. x.
    Uses Abramowitz & Stegun approximation to erf'''
    ss = np.sign(x)
    x = x*ss
    t = 1 / (1 + (0.47047/np.sqrt(2)*x))
    ee = np.exp(-0.5*x*x)
    e = 0.5 - 0.5* ee * t * ( 0.3480242 + t*(-0.0958798+t* 0.7478556))
    de = ee / np.sqrt(2*np.pi)
    d2e = -x * de
    return 0.5-e*ss, -de, -d2e*ss

def pSel(cov, moments, fluxMin):
    '''Calculate selection probability for templates with
    moments given by rows of `moment` given a flux cut at
    `fluxMin` and a measurement covariance of `cov` for the
    even moments.
    Covariance of the centroid moments is taken from the
    equivalent values in `cov`, which should be "even" matrix. '''
    # Here's the array that pops up in Jacobian
    B = np.array([0.25, -0.25, -0.25])
    m = bfd.Moment()
    # Calculate things needed from the (single) cov
    invSigF = 1. / np.sqrt(cov[m.M0,m.M0])
    A = 2 * B * cov[m.M0, m.MR:m.M2+1] * invSigF
    b = np.dot( cov[m.M0, m.MR:m.M2+1]**2, B) * invSigF * invSigF
    xyCov = 0.5 * np.array( [[ cov[m.M0,m.MR]+cov[m.M0,m.M1], cov[m.M0,m.M2] ],
                             [ cov[m.M0, m.M2], cov[m.M0,m.MR]-cov[m.M0,m.M1]]] )
    invXYCov = np.linalg.inv(xyCov)
    xyNorm = 1./np.sqrt(np.linalg.det(2 * np.pi * xyCov))
    
    # Now make calculations across the moments
    umin = invSigF * (fluxMin-moments[:,m.M0])
    y, dy, d2y = eFunc2(umin)
    jacobian = np.dot( moments[:,m.MR:m.M2+1]**2, B)
    out = jacobian  * y
    out -= np.dot(moments[:,m.MR:m.M2+1], A) * dy
    out += b * d2y
    
    # Now the xy likelihood term
    chisq = np.einsum('ij,ik,jk->i',moments[:,5:7],moments[:,5:7],invXYCov)
    out *= np.exp(-0.5*chisq) * xyNorm

    #for i in range(10):
    #    print(i,jacobian[i],y[i],np.dot(moments[i,m.MR:m.M2+1]**2, B*invSigF) * dy,
    #         np.exp(-0.5*chisq[i]) * xyNorm)
    return out
