import sys
import os
import math
import logging
import argparse
import time
import pdb

import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import astropy.io.fits as fits
import bfd

from scipy.optimize import minimize


def parse_input():

    '''Read command-line arguments and return object with parameter values
    '''
    parser = argparse.ArgumentParser(
        description='Determine best band weights for given set of input bands.\n',formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--input', '-i',
                            help='file name base for template moment files',
                            type=str,default='supplyname')
    parser.add_argument('--bands',
                        help='list of bands for which computing weights',
                        type=str,nargs='+',default='b')
    parser.add_argument('--moment',
                        help='tell which moment to use for weight calc, options are MF, MR, M1, M2, MC',
                        type=str,default='MF')
    parser.add_argument('--var',
                        help='user provides list of typical wide-field variances for particular moment choice',
                        type=float,nargs='+',default=0.0)

    args=parser.parse_args()
    return args

def check_pars(pars):

    if pars.input == 'supplyname':
        raise Exception("supply the correct file name base")

    if pars.bands == 'b':
        raise Exception("supply valid list of bands")
    
    valid_moment=["MF","MR","M1","M2","MC"]
    if pars.moment not in valid_moment:
        raise Exception("supply valid moment")

    if pars.var == 0:
        raise Exception("supply valid variance value")

    if len(pars.var) != len(pars.bands):
        raise Exception("list of variances must equal list of bands")


def get_norm_weight(w):
    return w/np.sum(w)

def sn_filter(mm,cc,dd,snmin,snmax):
    NN=np.shape(mm)[0]
    nn=np.shape(mm)[1]
    gd=np.zeros(NN)
    for i in xrange(NN):
        sn=np.abs(mm[i,:])/np.sqrt(cc[i,:])
        totgd=np.sum((sn > snmin) & (sn < snmax))
        if totgd==nn:
            gd[i]=1
    sel=np.where(gd)[0]
    return mm[sel],cc[sel],dd[sel]
                           
            
class sums_and_weights(object):
    def __init__(self,covariances,derivatives,nbands,maxval=1e10):
        self.c=covariances
        self.d=derivatives
        self.nbands=nbands
        self.bandsum=None
        self.galsum=None
        self.maxval=maxval
        return

    def sum_bands(self,w):
        if len(w) != self.nbands:
            raise Exception("weight array is not same as number of bands")
        dd=np.transpose(np.matrix(self.d))
        ww=np.matrix(w)
        ww2=np.matrix(w**2)
        cc=np.transpose(np.matrix(self.c))
        num=np.squeeze(np.array(np.dot(ww,dd)))**2
        den=np.squeeze(np.array(np.dot(ww2,cc)))
        self.bandsum=num/den
#        self.bandsum=1/den
        return self.bandsum

    def sum_gals(self):
        if self.bandsum is None:
            raise Exception("must sum bands before summing galaxies")
        aa=np.zeros((len(self.bandsum),2))
        aa[:,0]=self.maxval
        aa[:,1]=self.bandsum
        bandsum_mincut=np.min(aa,axis=1)
        self.galsum=np.sum(bandsum_mincut)
        return self.galsum

    def get_sum(self,w):
        aa=self.sum_bands(w)
        bb=self.sum_gals()
        return bb

    def get_sum_inverse(self,w):
        aa=self.sum_bands(w)
        bb=self.sum_gals()
        return 1./bb



    def con(self,t):
        return np.sum(t) - 1

         
    def min_sum(self,w):
        x0=w
        cons={'type':'eq','fun':self.con}
        bnds=((0,1),)*self.nbands
        out=minimize(self.get_sum_inverse,x0,constraints=cons,bounds=bnds,tol=1e-25,method='SLSQP')
#        out=minimize(self.get_sum,x0,bounds=bnds,method='COBYLA')
        return out

def get_ids_and_derivs(filenamebase,band,moment):
    x=fits.open(filenamebase+band+".fits")
    id_uniq=np.unique(x[1].data['id'])
    deriv_g1_uniq=[]
    deriv_g2_uniq=[]
    if moment == "MF":
        index=0
    if moment == "MR":
        index=1
    if moment == "M1":
        index=2
    if moment == "M2":
        index=3
    if moment == "MC":
        index=4
    
    for ii in xrange(len(id_uniq)):
        sel=np.where(x[1].data['id'] == id_uniq[ii])[0]
        deriv_g1_uniq.append(x[1].data['moments_dg1'][sel[0],index])
        deriv_g2_uniq.append(x[1].data['moments_dg2'][sel[0],index])

    deriv_g1_uniq=np.array(deriv_g1_uniq)
    deriv_g2_uniq=np.array(deriv_g2_uniq)
    return id_uniq, deriv_g1_uniq, deriv_g2_uniq

def get_matched_arrays(ids,dmdg1,dmdg2):
    numarrays=len(ids)
    for ii in xrange(numarrays-1):
        if ii == 0:
            ida=ids[0]
        else:
            ida=idfinal

        idb=ids[ii+1]
        idfinal=np.intersect1d(ida,idb,assume_unique=True)


    numtemp=len(idfinal)
    matched_dmdg1=np.zeros((numtemp,numarrays))
    matched_dmdg2=np.zeros((numtemp,numarrays))

    for ii in xrange(numarrays):
        idii=ids[ii]
        dmdg1ii=dmdg1[ii]
        dmdg2ii=dmdg2[ii]
        for jj in xrange(numtemp):
            aa=np.where(idii == idfinal[jj])[0]
            matched_dmdg1[jj,ii]=dmdg1ii[aa][0]
            matched_dmdg2[jj,ii]=dmdg2ii[aa][0]


    return matched_dmdg1,matched_dmdg2

def perform_minimization(dmdg, bands, var):
    #set up covariance arrays for calculation
    covars=dmdg*0.0
    for ii in xrange(len(bands)):
        covars[:,ii]=var[ii]

    #set up inital weights for number of bands
    ww=np.ones(len(bands))
    ww=get_norm_weight(ww)

    #set up sums and weights class
    SAW=sums_and_weights(covars,dmdg,len(bands))

    aa=SAW.sum_bands(ww)
    bb=SAW.sum_gals()


    out=SAW.min_sum(ww)
    return out.x

def main(pars):

    # check input parameters
    check_pars(pars)

    allids=[]
    allderivs_g1=[]
    allderivs_g2=[]
    for band in pars.bands:
        ids, derivs_g1, derivs_g2 = get_ids_and_derivs(pars.input, band, pars.moment)
        allids.append(ids)
        allderivs_g1.append(derivs_g1)
        allderivs_g2.append(derivs_g2)

    matched_derivs_g1,matched_derivs_g2 = get_matched_arrays(allids,allderivs_g1,allderivs_g2)

    weights_g1=perform_minimization(matched_derivs_g1, pars.bands, pars.var)
    weights_g2=perform_minimization(matched_derivs_g2, pars.bands, pars.var)

    print("Best weights using " + pars.moment + " wrt dg1:")
    for ii,band in enumerate(pars.bands):
        print("w_"+band + "= %s" %(weights_g1[ii]))
    print(" ")
    print("Best weights using " + pars.moment + " wrt dg2:")
    for ii,band in enumerate(pars.bands):
        print("w_"+band + "= %s" %(weights_g2[ii]))


    return
    


if __name__ == '__main__':

    pars=parse_input()

    main(pars)

'''
# MOST COMMON NUMBER OF EXPOSURES IS ~ 3
# Var MF r =  45578
# Var MF i = 147764
# Var MF z = 384624

# Var MR r =  75869
# Var MR i = 251325
# Var MR z = 603633

# Var M1 r =  38035
# Var M1 i = 126058
# Var M1 z = 302866

# Var M2 r =  37852
# Var M2 i = 125338
# Var M2 z = 300811

Using Flux Moment:
python ../../bfd/src/determine_band_weights.py --input bfd/moments --bands r i z --var 45578.0 147764.0 384624.0 --moment MF

Best weights using MF wrt dg1:
w_r= 0.535038077121
w_i= 0.304431720589
w_z= 0.160530202291

Best weights using MF wrt dg2:
w_r= 0.546050118319
w_i= 0.299520591613
w_z= 0.154429290068

Using R Moment
python ../../bfd/src/determine_band_weights.py --input bfd/moments --bands r i z --var 75869.0 251325.0 603633.0 --moment MR

Best weights using MR wrt dg1:
w_r= 0.520824068765
w_i= 0.304902947794
w_z= 0.174272983441
 
Best weights using MR wrt dg2:
w_r= 0.537365529451
w_i= 0.295423582849
w_z= 0.1672108877

Using M1 moment
python ../../bfd/src/determine_band_weights.py --input bfd/moments --bands r i z --var 38035.0 126058.0 302866.0 --moment M1

Best weights using M1 wrt dg1:
w_r= 0.625730871521
w_i= 0.249830121673
w_z= 0.124439006806
 
Best weights using M1 wrt dg2:
w_r= 0.556214431382
w_i= 0.30143225464
w_z= 0.142353313979

Using M2 Moment
python ../../bfd/src/determine_band_weights.py --input bfd/moments --bands r i z --var 37852.0 125338.0 300811.0 --moment M2
Best weights using M2 wrt dg1:
w_r= 0.555867825783
w_i= 0.301563396599
w_z= 0.142568777618

Best weights using M2 wrt dg2:
w_r= 0.616116898628
w_i= 0.25361136665
w_z= 0.130271734722
