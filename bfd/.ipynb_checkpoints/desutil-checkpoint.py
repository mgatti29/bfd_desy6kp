import sys
import os
import math
import logging
import argparse
import time
import pdb

import yaml
import numpy as np
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
import bfd
import bfdmeds
import ngmixer
import ngmix.gmix as gmix
import galsim
import pickle

def return_bandinfo(params):
    bandinfo={}
    bandweights=np.array(params['band_weights'],dtype='float32')
    bandinfo['bands']=params['bands']
    bandinfo['weights']=bandweights
    bandinfo['index']=np.arange(len(params['bands']))
    bands=np.array(params['bands'])
    return bands, bandinfo

def return_objinfo(filename,psf_source):
    meds0=bfdmeds.BFDMEDS(filename,psf_source)
    numcutouts = meds0._cat['ncutout']
    ids=meds0._cat['id']
    try:
        numbers=meds0._cat['number']
    except:
        numbers=ids
    ras=meds0._cat['ra']
    decs=meds0._cat['dec']
    numobjs = numcutouts.shape[0] 
    return numobjs, ids, numbers, ras, decs

def read_mof_file(filename):
    filenameparts=filename.split('.')
    # read in as fits file or pickle file
    # pull out mags column
    if filenameparts[-1]=='fits':
        mof=fits.open(filename)
        datamodels=mof[1].data
        dataepochs=mof[2].data
        cols=datamodels.columns.names
        opticalmodels=[]
    elif filenameparts[-1]=='p':
        datamodels=pickle.load(open(filename,"rb"))
        cols=datamodels.keys()
        filenameparts2=filename.split("_")
        mof_epoch_name=''
        for f in filenameparts2:
            if f !='models':
                add=f+"_"
            else:
                add="epochs_"
            mof_epoch_name=mof_epoch_name+add
        
        dataepochs=pickle.load(open(mof_epoch_name[0:-1],"rb"))
        if 'JHK' in filenameparts2:
            optical_model_name=''
            for f in filenameparts2:
                if f!='JHK':
                    add=f+"_"
                else:
                    add=""
                optical_model_name=optical_model_name+add
            opticalmodels=pickle.load(open(optical_model_name[0:-1],"rb"))
        else:
            opticalmodels=[]
    else:
        raise Exception("mof file type not fits or pickle")
    return datamodels,cols,dataepochs,opticalmodels

def get_g_minus_i(mof_file,meds_ids):
    data,cols,dataepochs,optical=read_mof_file(mof_file)
    if 'cm_mag' in cols:
        mags=data['cm_mag']
    elif 'bdf_mag' in cols:
        mags=data['bdf_mag']
    else:
        raise Exception("mof file must contain cm or bdf models")

    if np.shape(mags)[1]==4:gind,iind=0,2 #griz
    elif np.shape(mags)[1]==5:gind,iind=1,3 #ugriz
    else:
        return np.ones(np.shape(mags)[0]) # if nonstandard, just return ones
    
    mof_g_minus_i = mags[:,gind]-mags[:,iind]
    mof_ids=data['id']
    # sort both meds and mof ids
    aa=np.argsort(meds_ids)
    bb=np.argsort(mof_ids)
    # created sorted lists
    meds_ids_check=meds_ids[aa]
    mof_ids_check=mof_ids[bb]
    mof_g_minus_i_sort=mof_g_minus_i[bb]
    # transform sorted lists back to meds ordering
    meds_ids_check2=meds_ids_check[aa]
    mof_g_minus_i_sortmeds=mof_g_minus_i_sort[aa]

    return mof_g_minus_i_sortmeds

def perform_mof_correction(mof_file,band,start,end,params):
    # save mof corrected meds file for particular chunk of objects to temporary dir
    infilenew=os.path.join(params['dir'],'tempdir',params['infile'].split('/')[-1] + '_' + band + '_meds_mofcorr_'+str(start)+'_'+str(end)+'.fits.fz')
    fileexists = os.path.isfile(infilenew)
    if fileexists==False:
        mof=ngmixer.imageio.extractor_corrector.MEDSExtractorCorrector( 
            mof_file,
            os.path.join(params['dir'],params['infile']+'_'+band+'_meds-'+params['infile_version']+'.fits.fz'),
            start,
            end,
            infilenew,
            reject_outliers=True,
            replace_bad=True,
            verbose=False,
            bad_flags=[
                1,     # MAKE SURE SET
                2,     # SATURATE
                4,     # INTERP
                2048,  # SUSPECT
                8192,  # NEAREDGE
                16384, # TAPEBUMP
            ])
    else:
        print("file already exists, yay")

def return_meds_list(bands,psf_source,astro_dir,mof_file,params,start=0,end=0,colors=None):
    meds=[]
    medsorig=[]
    
    if (mof_file is not None) & (colors is None):
        raise Exception("Must supply colors if giving mof_file")
    for j in xrange(len(bands)):
        if mof_file is None:
            # no mof file supplied, default to plain color
            meds.append(bfdmeds.BFDMEDS(os.path.join(params['dir'],
                                                     params['infile']+'_'+bands[j]+'_meds-'+params['infile_version']+'.fits.fz'),
                                        psf_source,
                                        astro_dir = astro_dir))
            medsorig=None
        else:
            # mof file supplied, give objects g-i color
            colors_sub = colors[start:end+1]
            if params['mof_subtraction']=="ngmixer":
                meds.append(bfdmeds.BFDMEDS(os.path.join(params['dir'],'tempdir',
                                                         params['infile'].split('/')[-1]+'_'+bands[j]+'_meds_mofcorr_'+str(start)+'_'+str(end)+'.fits.fz'),
                                            psf_source,astro_dir = astro_dir,color_array=colors_sub))
                medsorig.append(bfdmeds.BFDMEDS(os.path.join(params['dir'],
                                                             params['infile']+'_'+bands[j]+'_meds-'+params['infile_version']+'.fits.fz'),
                                                psf_source,astro_dir = astro_dir,color_array=colors))
            elif params['mof_subtraction']=="internal":
                meds.append(bfdmeds.BFDMEDS(os.path.join(params['dir'],
                                                             params['infile']+'_'+bands[j]+'_meds-'+params['infile_version']+'.fits.fz'),
                                                psf_source,astro_dir = astro_dir,color_array=colors))
                medsorig=None
            else:
                raise Exception("if supplying mof file must supply valid subtraction routine (ngmixer or internal)")

    return meds,medsorig


def remeasure_background(objim,noise,nbrim=None):

    maskarr=np.ones(np.shape(objim),dtype='int')
    maskarr[3:-3,3:-3]=0
    if nbrim is not None:
        sel=np.where(nbrim > 3.0*noise)
        maskarr[sel]=0

    return np.mean(objim[np.where(maskarr==1)]),np.std(objim[np.where(maskarr==1)]),np.sum(maskarr==1)


def get_nbr_info(centralid, datamodel,dataepoch,boxsize,modeltype,pixelscale):
    # get galaxy in mof file
    mof_cent_index=np.where(datamodel['id']==centralid)[0]
    # find separations of all galaxies to central
    all_coord = SkyCoord(ra=datamodel['ra']*u.degree,dec=datamodel['dec']*u.degree)
    cent_coord=all_coord[mof_cent_index[0]]
    sep=cent_coord.separation(all_coord)
    maxsep=boxsize*(pixelscale/2.0)*np.sqrt(2.0)*1.05
    # define neighbors within certain separation
    mof_nbr_indices=np.where(sep < maxsep*u.arcsec)[0]

    nbrid=[]
    nbrgdfit=[]
    nbrfileid=[]
    nbrmodelindex=[]
    nbrepochindex=[]
    nbrbandindex=[]

    for i in xrange(len(mof_nbr_indices)):
        # skip central
        if datamodel['id'][mof_nbr_indices][i] == centralid:
            continue
        epoch_nbr_indices=np.where(dataepoch['id']==datamodel['id'][mof_nbr_indices][i])[0]
        for j in xrange(len(epoch_nbr_indices)):
            nbrid.append(datamodel['id'][mof_nbr_indices][i])
            nbrgdfit.append((datamodel['flags'][mof_nbr_indices][i]==0))# & (datamodel['mask_flags'][mof_nbr_indices][i]==0))
            nbrmodelindex.append(mof_nbr_indices[i])
            nbrfileid.append(dataepoch['file_id'][epoch_nbr_indices][j])
            nbrepochindex.append(epoch_nbr_indices[j])
            if modeltype=='cm':
                nbrbandindex.append(dataepoch['band_num'][epoch_nbr_indices][j])
            if modeltype=='bdf':
                nbrbandindex.append(dataepoch['band'][epoch_nbr_indices][j])
    return nbrid,nbrgdfit,nbrfileid,nbrmodelindex,nbrepochindex,nbrbandindex



def return_band_val(bandind,numbands,indtostr=True):
    if indtostr:
        if numbands==5:
            if bandind==0: return 'u'
            if bandind==1: return 'g'
            if bandind==2: return 'r'
            if bandind==3: return 'i'
            if bandind==4: return 'z'
        if numbands==4:
            if bandind==0: return 'g'
            if bandind==1: return 'r'
            if bandind==2: return 'i'
            if bandind==3: return 'z'
        if numbands==3:
            if bandind==0: return 'J'
            if bandind==1: return 'H'
            if bandind==2: return 'Ks'
    else:
        if numbands==5:
            if bandind=='u': return 0
            if bandind=='g': return 1
            if bandind=='r': return 2
            if bandind=='i': return 3
            if bandind=='z': return 4
        if numbands==4:
            if bandind=='g': return 0
            if bandind=='r': return 1
            if bandind=='i': return 2
            if bandind=='z': return 3
        if numbands==3:
            if bandind=='J':  return 0
            if bandind=='H':  return 1
            if bandind=='Ks': return 2

            
    
def render_gal(gal_pars,psf_pars,wcs,shape,model='bdf',debug=False):
 
    psf_gmix=gmix.GMix(pars=psf_pars)
    e1,e2,T=psf_gmix.get_e1e2T()

    det=np.abs(wcs.getdet()) 
    jac=gmix.Jacobian(row=wcs.xy0[1],
                      col=wcs.xy0[0],
                      dudrow=wcs.jac[0,1],
                      dudcol=wcs.jac[0,0],
                      dvdrow=wcs.jac[1,1],
                      dvdcol=wcs.jac[1,0])

    if model=='bdf':
        gmix_sky = gmix.GMixBDF(gal_pars)
    elif model=='cm':
        # fracdev index is 6
        # tdbyte index is 7
        # normal pars are 0-5
        gmix_sky = gmix.GMixCM(gal_pars[6],gal_pars[7],gal_pars[0:6])
    else:
        raise Exception("must supply valid model (bdf or cm)")

    gmix_image = gmix_sky.convolve(psf_gmix)
    try:
        image = gmix_image.make_image((shape,shape), jacobian=jac, fast_exp=True)
    except:
        image=np.zeros((shape,shape))

    return image*det

def trim_psf(obj,psf):
    # object and psf are not the same size
    # if object is larger, zeropad psf
    # if object is smaller, cut psf to same size
    dims_obj,dims_psf=np.shape(obj),np.shape(psf)
    # check both square
    if dims_obj[0] != dims_obj[1]:
        raise Exception("Object pstage stamp not square")
    if dims_psf[0] != dims_psf[1]:
        raise Exception("PSF postage stamp not square")

    if dims_obj[0] < dims_psf[0]:
        # object smaller than psf, cut out smaller PS around PSF
        if dims_psf[0]%2 == 0:
            dim1=dims_psf[0]/2
            dim2=dims_obj[0]/2
        else:
            dim1=(dims_psf[0]+1)/2
            dim2=dims_obj[0]/2
            
        psf_return=psf[dim1-dim2:dim1+dim2,dim1-dim2:dim1+dim2]
    else:
 
        # object larger than psf, zeropad
        psf_return=np.zeros(dims_obj)
        dim1=dims_obj[0]/2
        if dims_psf[0]%2==0:
            dim2=dims_psf[0]/2
            psfdim2=0
        else:
            dim2=(dims_psf[0]-1)/2
            psfdim2=1
            
        psf_return[dim1-dim2:dim1+dim2,dim1-dim2:dim1+dim2]=psf[psfdim2:,psfdim2:]

    return psf_return

def return_kdata_list(ii,bands,meds,params,medsorig=None,offset=0,skip_coadd=True,mof_info=None):
    # ii is theindex of the object to returna kdata list for
    # if no ngmixer mof correction has been done, then offset=0 and the indexing proceeds as normal
    # if ngmixer mof correction has been done, then offset=start (since the ngmixer mof corrected meds file only contains this subset of objects)
    # if internal mof correction to be done, will be done on the fly here
    # skip_coadd specified in parameters, automatically set to False for making templates or the band weight test file
    # psf_source is the source of the PSFs, if None will just be arrays from the medsfile itself

    fitsfilecheck=False
    kdata=[]
    kdata_bkgdsub=[]
    kdata_resid=[]
    if params['mof_subtraction']=="internal":
        if mof_info is None:
            raise Exception("need to supply mof file for internal mof subtraction")
        # find neighbors
        if 'cm_mag' in mof_info[1]:
            modeltype='cm'
        elif 'bdf_mag' in mof_info[1]:
            modeltype='bdf'
        else:
            raise Exception("must supply valid model type (cm or bdf)")
        # get pixelscale
        wcs=meds[0].get_jacobian_list(ii,skip_coadd=skip_coadd)[0]
        jac=np.array([[wcs['dudcol'],wcs['dudrow']],[wcs['dvdcol'],wcs['dvdrow']]])
        pixelscale=np.sqrt(np.abs(np.linalg.det(jac)))
        nbrid,nbrgd,nbrfileid,nbrmodelindex,nbrepochindex,nbrbandindex=get_nbr_info(meds[0]['id'][ii],mof_info[0],mof_info[2],meds[0]['box_size'][ii],modeltype,pixelscale)


    if fitsfilecheck:
        objsave=[]
        origobjsave=[]
        row0save=[]
        col0save=[]
    # loop through bands
    for j in xrange(len(bands)):
        medsj=meds[j]
        # get image, weight, and bpm stamps for each band 
        objij=medsj.get_cutout_list(ii-offset,skip_coadd=skip_coadd)
        wmapij=medsj.get_cutout_list(ii-offset,'weight',skip_coadd=skip_coadd)
        segij=medsj.get_cutout_list(ii-offset,'seg',skip_coadd=skip_coadd)
        fileidij=medsj['file_id'][ii-offset]
        if skip_coadd:
            fileidij=fileidij[1:]

        # not always a bad pixel map (for image sims)
        try:
            bmapij=medsj.get_cutout_list(ii-offset,'bmask',skip_coadd=skip_coadd)
        except:
            bmapij=[np.zeros(np.shape(objij[0])) for _ in objij]
        
        # get image for original image if mof_file is provided
        if medsorig is not None:
            medsorigj=medsorig[j]
            origobjij=medsorigj.get_cutout_list(ii,skip_coadd=skip_coadd)

        # get list of wcs - need to replace with pixmappy, is it okay for coadd? 
        # (meds saves psf from y,x vs. x,y position of detector)
        wcsij = medsj.get_jacobian_list(ii-offset, skip_coadd=skip_coadd)
 
        # get list of background noise for each image/band
        noiseij = medsj.get_noise_list(ii-offset, skip_coadd=skip_coadd)

        # get PSF for each image/band, returned as list of images
        psfij = medsj.get_psf_list(ii-offset,skip_coadd=skip_coadd)

        if fitsfilecheck:
            objsave.append(objij[0])
            origobjsave.append(origobjij[0])
            row0save.append(wcsij[0]['row0'])
            col0save.append(wcsij[0]['col0'])
        # get number of images for band and loop through
        nims = len(objij)
        for k in range(nims):
            # assess number of 0 weighted pixels in bpm
            # if below given threshold, include in kdata list
            # specify bpm values that set to 0 and corrected in ngmixer step
            
            #badpix=(bmapij[k] == 1) | (bmapij[k] == 2) | (bmapij[k] == 4) | (bmapij[k] == 8) | \
            #    (bmapij[k] == 16) | (bmapij[k] == 64) | (bmapij[k] == 128) | (bmapij[k] == 256) | \
            #    (bmapij[k] == 512) | (bmapij[k] == 1024) | (bmapij[k] == 2048) | (bmapij[k] == 8192) | \
            #    (bmapij == 16384)
            
            badpix = (bmapij[k] != 0) | (wmapij[k] == 0)
            bad_pixel_percentage = np.float(np.sum(badpix))/np.float(np.size(wmapij[k]))

            if bad_pixel_percentage <= params['bad_pixel_threshold']:
                # define WCS
                origin = (0.,0.)                             # "World" coord galaxy center
                xyref = (wcsij[k]['col0'],wcsij[k]['row0'])  # pixel coordinate of fixed point
                uvref = (0.,0.)                              # world coordinates of fixed point
                duv_dxy = np.array( [ [wcsij[k]['dudcol'], wcsij[k]['dudrow']],
                                      [wcsij[k]['dvdcol'], wcsij[k]['dvdrow']] ])
                wcs = bfd.WCS(duv_dxy, xyref=xyref, uvref=uvref)
                wcsgalsim=galsim.AffineTransform(wcsij[k]['dudcol'],wcsij[k]['dudrow'],wcsij[k]['dvdcol'],wcsij[k]['dvdrow'],origin=galsim.PositionD(xyref[0],xyref[1]),world_origin=galsim.PositionD(0,0))

                if params['mof_subtraction']=="internal":
                    nbr_im=np.zeros(np.shape(objij[k]))
                    if modeltype=='cm': numbands=np.shape(mof_info[0]['cm_mag'])[1]
                    if modeltype=='bdf':numbands=np.shape(mof_info[0]['bdf_mag'])[1]
                    for nid,ngd,nfileid,nmodelind,nepochind,nbandind in zip(nbrid,nbrgd,nbrfileid,nbrmodelindex,nbrepochindex,nbrbandindex):
                        if (return_band_val(nbandind,numbands) != bands[j]): continue
                        if ngd==False: continue
                        if nfileid != fileidij[k]: continue
                        if mof_info[0]['bdf_flux'][nmodelind][nbandind] < 0.0: continue

                        if modeltype=='cm':
                            # orig 5, flux, fracdev, tdbyte
                            gal_pars=mof_info[0]['cm_pars'][nmodelind][0:5] 
                            gal_pars=np.append(gal_pars,mof_info[0]['cm_flux'][nmodelind][nbandind])
                            gal_pars=np.append(gal_pars,mof_info[0]['cm_fracdev'][nmodelind])
                            gal_pars=np.append(gal_pars,mof_info[0]['cm_TdbyTe'][nmodelind])
                        elif modeltype=='bdf':
                            # defaultts to optical model params if specified
                            if len(mof_info[3])>0:
                                aa=np.where(mof_info[3]['id']==mof_info[0]['id'][nmodelind])[0]
                                if len(aa)>0:
                                    gal_pars=mof_info[3]['bdf_pars'][aa[0]][0:6]
                                else:
                                    continue
                            else:
                                # 6 params + flux
                                gal_pars=mof_info[0]['bdf_pars'][nmodelind][0:6]
                            gal_pars=np.append(gal_pars,mof_info[0]['bdf_flux'][nmodelind][nbandind])
                            if gal_pars[0] < -999:
                                continue
                        else:
                            raise Exception("invalid model supplied")
                        if modeltype=='cm':
                            psf_pars=mof_info[2]['psf_fit_pars'][nepochind]
                        if modeltype=='bdf':
                            psf_pars=mof_info[2]['psf_pars'][nepochind]
                        medind=np.where(medsj['id']==nid)[0][0]
                        fileind=np.where(medsj['file_id'][medind]==fileidij[k])[0][0]
                        mainind=k
                        if skip_coadd:
                            mainind=k+1
                        nbrwcs = medsj.get_jacobian_list(medind, skip_coadd=False)
                        nbrrowisy = medsj['cutout_row'][ii][mainind] - (medsj['orig_row'][ii][mainind]-medsj['orig_row'][medind][fileind])
                        nbrcolisx = medsj['cutout_col'][ii][mainind] - (medsj['orig_col'][ii][mainind]-medsj['orig_col'][medind][fileind])
                        nbrxyref = (nbrcolisx,nbrrowisy)
                        nbruvref = (0,0)
                        nbrduv_dxy = np.array( [ [nbrwcs[fileind]['dudcol'], nbrwcs[fileind]['dudrow']],
                                                 [nbrwcs[fileind]['dvdcol'], nbrwcs[fileind]['dvdrow']] ])
                        nbrwcs = bfd.WCS(nbrduv_dxy, xyref=nbrxyref, uvref=nbruvref)
                        nbr_im+=render_gal(gal_pars,psf_pars,nbrwcs,np.shape(objij[k])[0],model=modeltype)

                imuse=None
                imusebkgdsub=None
                imuseresid=None
                psfuse=None
                if params['mof_subtraction']=='none':
                    imuse=objij[k]
                    bkgd_estimate,bkgd_uncertainty,bkgd_npix=remeasure_background(objij[k],noiseij[k])
                if params['mof_subtraction']=='ngmixer':
                    imuse=objij[k]
                    bkgd_estimate,bkgd_uncertainty,bkgd_npix=remeasure_background(objij[k],noiseij[k],nbrim=origobjij[k]-objij[k])
                    imuseresid=origobjij[k]-objij[k]
                if params['mof_subtraction']=='internal':
                    imuse=objij[k]-nbr_im
                    bkgd_estimate,bkgd_uncertainty,bkgd_npix=remeasure_background(objij[k],noiseij[k],nbrim=nbr_im)
                    imuseresid=nbr_im
                if medsj.psf_source==None:
                    psfuse=psfij[k]
                else:
                    psfuse=psfij[k].array
                if np.shape(psfuse)[0] != np.shape(imuse)[0]:
                    psfuse=trim_psf(imuse,psfuse)

                kdata.append( bfd.simpleImage(imuse, origin,psfuse, wcs=wcs, pad_factor=params['pad_factor'], pixel_noise=noiseij[k],band=bands[j]))
                if imuseresid is not None:
                    # get Fourier space image of residual (orig-mofcorr)
                    kdata_resid.append( bfd.simpleImage(imuseresid, origin, psfuse, wcs=wcs, pad_factor=params['pad_factor'],pixel_noise=noiseij[k],band=bands[j]))


    if ((fitsfilecheck) & (ii%120==5) & (ii > 0)):
        filename=os.path.join(params['dir'],'checkmofcorr3.fits')
        fileexists=os.path.isfile(filename)
        if fileexists:
            hdu=fits.open(filename)
        else:
            hdu=fits.PrimaryHDU()
            hdu.header.set('nbands',len(bands))
            for i in xrange(len(bands)):
                hdu.header.set('band'+str(i),bands[i])

        imdimsave=[]
        for i in xrange(len(objsave)):
            imdim=np.shape(objsave[i])
            imdimsave.append(imdim[0])

        imdimmax=max(imdimsave)
        imagesave=np.zeros((imdimmax*4,imdimmax*2))

        for i in xrange(len(objsave)):
            imagesave[i*imdimmax:i*imdimmax+imdimsave[i],0:imdimsave[i]]=origobjsave[i]
            imagesave[i*imdimmax:i*imdimmax+imdimsave[i],imdimsave[i]:2*imdimsave[i]]=objsave[i]

            #imagesave[i*imdimmax,0:imdimsave[i]*2]=1
            #imagesave[i*imdimmax+imdimsave[i]-1,0:imdimsave[i]*2]=1
            #imagesave[i*imdimmax:i*imdimmax+imdimsave[i],0]=1
            #imagesave[i*imdimmax:i*imdimmax+imdimsave[i],2*imdimsave[i]-1]=1


        hdunew=fits.ImageHDU(imagesave)
        hdunew.header.set('ID',medsj['id'][ii])
        hdunew.header.set('radius',medsj['iso_radius'][ii])
        for i in xrange(len(bands)):
            hdunew.header.set('dim'+str(i),imdimsave[i])
            hdunew.header.set('col0'+str(i),col0save[i])
            hdunew.header.set('row0'+str(i),row0save[i])
        if fileexists:
            hdu.append(hdunew)
            newhdu=hdu
        else:
            newhdu=fits.HDUList([hdu,hdunew])
        newhdu.writeto(filename,overwrite=True)


    return kdata,kdata_resid



def add_entries(taborig,tabadd,template=False,band_weight_file=False):

    if band_weight_file == False:
        if template == False:
            taborig.id.extend(tabadd.id)
            taborig.moment.extend(tabadd.moment)
            taborig.xy.extend(tabadd.xy)
            taborig.number.extend(tabadd.number)
            taborig.num_exp.extend(tabadd.num_exp)
            taborig.cov_even.extend(tabadd.cov_even)
            taborig.cov_odd.extend(tabadd.cov_odd)
            taborig.delta_flux_moment.extend(tabadd.delta_flux_moment)
            taborig.cov_delta_flux_moment.extend(tabadd.cov_delta_flux_moment)
            taborig.nlost+=tabadd.nlost
        else:
            taborig.templates.extend(tabadd.templates)

    else:
        taborig.id.extend(tabadd.id)
        taborig.covariance.extend(tabadd.covariance)
        taborig.moment.extend(tabadd.moment)
        taborig.derivative.extend(tabadd.derivative)
        
    return taborig

def combine_tables(tabs,num_tabs,bands,template=False):
    
    # pull out first table
    tabs_first=tabs[0]
    tab=tabs_first[0]
    tab_resid=tabs_first[1]
    tab_band=tabs_first[2]
    tab_band_weight=tabs_first[3]

    # if more than one, concatenate other tables to end before saving out
    if num_tabs > 1:
        for jj in xrange(1,num_tabs):
            #pull out next set of tables
            tabs_next=tabs[jj]
            # add to main table if exists
            if tab is not None:
                tab=add_entries(tab,tabs_next[0],template=template)
            # add to residuals table if exists
            if tab_resid is not None:
                tab_resid=add_entries(tab_resid,tabs_next[1],template=template)
            # add to separate band tables if exists
            if len(tab_band) > 0:
                for kk in xrange(len(bands)):
                    tab_band[kk]=add_entries(tab_band[kk],tabs_next[2][kk],template=template)
            # add to the file for determining band weights if exist
            if tab_band_weight is not None:
                tab_band_weight=add_entries(tab_band_weight,tabs_next[3],band_weight_file=True)

    return tab,tab_resid,tab_band,tab_band_weight
