import sys
import os
import argparse
import meds as medsio
import numpy as np
import astropy.io.fits as fits
import multiprocessing
import copy
import time
import gc
import pdb

def parse_input():
    '''Read command-line arguments and
       return dictionary of parameter values.
    '''
    parser = argparse.ArgumentParser(
        description='Define neighbors on MEDS postage stamps and write new nbr_extension to copy of mof file.',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Config file names - do not get passed back to program.
    parser.add_argument('--medsfile', help='Input MEDS file', type=str,default=None)
    parser.add_argument('--moffile', help='Input mof file', type=str,default=None)
    parser.add_argument('--dir', help='Directory for files', type=str,default='./')
    parser.add_argument('--num_proc',help='number of processors to use',type=int,default=1)
    args = parser.parse_args()

    return args

def check_input(params):
    '''code to check input is okay'''

    if params.medsfile is None:
        raise Exception("Must specify meds file name")

    if os.path.isfile(os.path.join(params.dir,params.medsfile)) == False:
        raise Exception("given MEDS file does not exist, check directory and name")

    if params.moffile is None:
        raise Exception("Must specify mof file name")

    if os.path.isfile(os.path.join(params.dir,params.moffile)) == False:
        raise Exception("given mof file does not exist, check directory and name")

    return

def gen_neighbor_list(medsfile):
    '''code to generate the list of all neighbors on a meds file 
    using the segmentation map'''

    # setup lists to hold id of central galaxy and ids of neighboring galaxies
    main_list=[]
    nbr_list=[]

    # open up meds file
    meds = medsio.MEDS(medsfile)
    numobjs =  len(meds['id'])

    # loop over all objects to find all neighboring objects on a given frame using
    # the segmentation map
    for i in xrange(numobjs):
        # get central galaxy id
        mainid = meds['number'][i]
        # pull out segmenation map and find unique values
        seg_map = (meds.get_cutout_list(i,'seg'))[0]
        seg_uniq=np.unique(seg_map)
        # loop over unique segmentation values skipping 0 and central object id
        # save central id to main list and nbr id to nbr list if nbrs exist
        for j in xrange(len(seg_uniq)):
            if seg_uniq[j] == 0:
                continue
            if seg_uniq[j] == mainid:
                continue
            main_list.append(mainid)
            nbr_list.append(seg_uniq[j])

    # return the list of centrals, nbrs, and total number of objects
    return main_list, nbr_list, numobjs

def gen_neighbor_info(params):
    '''code to generate a dictionary to populate the new neighbor extension in the
    mof file'''
    
    # open mof file and get main, epoch, and neighbor extensions
    mof=fits.open(params['MOFFILE'])
    mof_main=mof[1]
    mof_epoch=mof[2]
    mof_nbr=mof[3]

    # get ids of all gals in file, and list of main/neighbor gals
    ids=mof_main.data['number']
    ngals=len(ids)
    
    main_gal_list=params['main_gal_list']
    nbr_gal_list=params['nbr_gal_list']

    # setup lists to save all data needed for new extension
    nbrinfo_id=[]
    nbrinfo_number=[]
    nbrinfo_band=[]
    nbrinfo_band_num=[]
    nbrinfo_cutout_index=[]
    nbrinfo_orig_row=[]
    nbrinfo_orig_col=[]
    nbrinfo_cutout_row=[]
    nbrinfo_cutout_col=[]
    nbrinfo_file_id=[]
    nbrinfo_dudrow=[]
    nbrinfo_dudcol=[]
    nbrinfo_dvdrow=[]
    nbrinfo_dvdcol=[]
    nbrinfo_pixel_scale=[]
    nbrinfo_image_id=[]
    nbrinfo_nbr_id=[]
    nbrinfo_nbr_flags=[]
    nbrinfo_nbr_jac_row0=[]
    nbrinfo_nbr_jac_col0=[]
    nbrinfo_nbr_jac_dudrow=[]
    nbrinfo_nbr_jac_dudcol=[]
    nbrinfo_nbr_jac_dvdrow=[]
    nbrinfo_nbr_jac_dvdcol=[]
    nbrinfo_nbr_psf_fit_pars=[]

    # loop over galaxies in range set by parameters
    for i in xrange(params['start'],params['end']+1):
        mainid=ids[i]
        if i%100==0:
            print("loop over %s gal" %(i))
        # check for neighbors
        sel=np.where(main_gal_list == mainid)[0]
        if len(sel)==0:
            continue

        # rely on mof_epoch file for epoch information
        aa=np.where(mof_epoch.data['number']==mainid)[0]

        # if main gal does not appear in epoch info, then skip
        if len(aa)==0:
            continue

        # how many neighbors are there?
        num_nbr=len(sel)
        ab=time.clock()

        # loop over bands (4 bands)
        for j in xrange(4):
            band='blah'
            if j==0:
                band='g'
            if j==1:
                band='r'
            if j==2:
                band='i'
            if j==3:
                band='z'

            # loop over neighbors + main galaxy
            for k in xrange(num_nbr+1):
                # neighbor k is:
                # have to include the original object in the final spot
                if k==num_nbr:
                    nbr_k_num=mainid
                else:
                    nbr_k_num = np.array(nbr_gal_list)[sel][k]

                # get index of nbr in main and epoch extensions
                fndm=np.where(mof_main.data['number']==nbr_k_num)[0]
                fnde=np.where(mof_epoch.data['number']==nbr_k_num)[0]

                # if neighbor not in epoch extension or cmT is ludicrous
                # then skip
                if ((len(fnde)==0) | (mof_main.data['cm_T'][fndm] > 100)) & (k < num_nbr):
                    continue

                # info to get from main extension
                nbrinfo_id.append(mof_main.data['id'][i])
                nbrinfo_number.append(mof_main.data['number'][i])
                nbrinfo_band.append(band)
                nbrinfo_band_num.append(j)

                # info for main gal to get from epoch extension
                nbrinfo_cutout_index.append(0)
                nbrinfo_orig_row.append(mof_epoch.data['orig_row'][aa][j])
                nbrinfo_orig_col.append(mof_epoch.data['orig_col'][aa][j])
                nbrinfo_cutout_row.append(mof_epoch.data['cutout_row'][aa][j])
                nbrinfo_cutout_col.append(mof_epoch.data['cutout_col'][aa][j])
                nbrinfo_file_id.append(mof_epoch.data['file_id'][aa][j])
                nbrinfo_dudrow.append(mof_epoch.data['dudrow'][aa][j])
                nbrinfo_dudcol.append(mof_epoch.data['dudcol'][aa][j])
                nbrinfo_dvdrow.append(mof_epoch.data['dvdrow'][aa][j])
                nbrinfo_dvdcol.append(mof_epoch.data['dvdcol'][aa][j])
                nbrinfo_pixel_scale.append(mof_epoch.data['pixel_scale'][aa][j])
                nbrinfo_image_id.append(mof_epoch.data['image_id'][aa][j])

                # info for nbr gals to get from epoch extension           
                nbrinfo_nbr_id.append(mof_main.data['id'][fndm][0])
                nbrinfo_nbr_flags.append(0)#mof_main.data['cm_max_flags'][fndm][0])
                nbrinfo_nbr_jac_row0.append(mof_epoch.data['cutout_row'][aa][j]-(mof_epoch.data['orig_row'][aa][j]-mof_epoch.data['orig_row'][fnde][j]))
                nbrinfo_nbr_jac_col0.append(mof_epoch.data['cutout_col'][aa][j]-(mof_epoch.data['orig_col'][aa][j]-mof_epoch.data['orig_col'][fnde][j]))
                nbrinfo_nbr_jac_dudrow.append(mof_epoch.data['dudrow'][fnde][j])
                nbrinfo_nbr_jac_dudcol.append(mof_epoch.data['dudcol'][fnde][j])
                nbrinfo_nbr_jac_dvdrow.append(mof_epoch.data['dvdrow'][fnde][j])
                nbrinfo_nbr_jac_dvdcol.append(mof_epoch.data['dvdcol'][fnde][j])
                nbrinfo_nbr_psf_fit_pars.append(mof_epoch.data['psf_fit_pars'][fnde][j])

    # build nbr_info dictionary
    nbr_info={'id':np.array(nbrinfo_id),
              'number':np.array(nbrinfo_number),
              'band':np.array(nbrinfo_band),
              'band_num':np.array(nbrinfo_band_num),
              'cutout_index':np.array(nbrinfo_cutout_index),
              'orig_row':np.array(nbrinfo_orig_row),
              'orig_col':np.array(nbrinfo_orig_col),
              'cutout_row':np.array(nbrinfo_cutout_row),
              'cutout_col':np.array(nbrinfo_cutout_col),
              'file_id':np.array(nbrinfo_file_id),
              'dudrow':np.array(nbrinfo_dudrow),
              'dudcol':np.array(nbrinfo_dudcol),
              'dvdrow':np.array(nbrinfo_dvdrow),
              'dvdcol':np.array(nbrinfo_dvdcol),
              'pixel_scale':np.array(nbrinfo_pixel_scale),
              'image_id':np.array(nbrinfo_image_id),
              'nbr_id':np.array(nbrinfo_nbr_id),
              'nbr_flags':np.array(nbrinfo_nbr_flags),
              'nbr_jac_row0':np.array(nbrinfo_nbr_jac_row0),
              'nbr_jac_col0':np.array(nbrinfo_nbr_jac_col0),
              'nbr_jac_dudrow':np.array(nbrinfo_nbr_jac_dudrow),
              'nbr_jac_dudcol':np.array(nbrinfo_nbr_jac_dudcol),
              'nbr_jac_dvdrow':np.array(nbrinfo_nbr_jac_dvdrow),
              'nbr_jac_dvdcol':np.array(nbrinfo_nbr_jac_dvdcol),
              'nbr_psf_fit_pars':np.array(nbrinfo_nbr_psf_fit_pars)}

    return nbr_info

def combine_dictionaries(dicts,numproc):
    ''' code to combine all dictionaries from different processors into one'''
    # make copy of first dictionary
    dictfinal=copy.deepcopy(dicts[0])

    # loop over remaining dictionaries to add entries at end
    for ii in xrange(1,numproc):
        for key in dictfinal:
            dictfinal[key]=np.append(dictfinal[key],dicts[ii][key],axis=0)

    return dictfinal

def create_new_extension(nbr_info):
    '''code to create a new neighbor extension for the mof file'''

    c1=fits.Column(name="id",format="K",array=nbr_info['id'])
    c2=fits.Column(name="number",format="J",array=nbr_info['number'])
    c3=fits.Column(name="band",format="1A",array=nbr_info['band'])
    c4=fits.Column(name="band_num",format="I",array=nbr_info['band_num'])
    c5=fits.Column(name="cutout_index",format="J",array=nbr_info['cutout_index'])
    c6=fits.Column(name="orig_row",format="D",array=nbr_info['orig_row'])
    c7=fits.Column(name="orig_col",format="D",array=nbr_info['orig_col'])
    c8=fits.Column(name="cutout_row",format="D",array=nbr_info['cutout_row'])
    c9=fits.Column(name="cutout_col",format="D",array=nbr_info['cutout_col'])
    c10=fits.Column(name="file_id",format="J",array=nbr_info['file_id'])
    c11=fits.Column(name="dudrow",format="D",array=nbr_info['dudrow'])
    c12=fits.Column(name="dudcol",format="D",array=nbr_info['dudcol'])
    c13=fits.Column(name="dvdrow",format="D",array=nbr_info['dvdrow'])
    c14=fits.Column(name="dvdcol",format="D",array=nbr_info['dvdcol'])
    c15=fits.Column(name="pixel_scale",format="D",array=nbr_info['pixel_scale'])
    c16=fits.Column(name="image_id",format="49A",array=nbr_info['image_id'])
    c17=fits.Column(name="nbr_id",format="K",array=nbr_info['nbr_id'])
    c18=fits.Column(name="nbr_flags",format="J",array=nbr_info['nbr_flags'])
    c19=fits.Column(name="nbr_jac_row0",format="D",array=nbr_info['nbr_jac_row0'])
    c20=fits.Column(name="nbr_jac_col0",format="D",array=nbr_info['nbr_jac_col0'])
    c21=fits.Column(name="nbr_jac_dudrow",format="D",array=nbr_info['nbr_jac_dudrow'])
    c22=fits.Column(name="nbr_jac_dudcol",format="D",array=nbr_info['nbr_jac_dudcol'])
    c23=fits.Column(name="nbr_jac_dvdrow",format="D",array=nbr_info['nbr_jac_dvdrow'])
    c24=fits.Column(name="nbr_jac_dvdcol",format="D",array=nbr_info['nbr_jac_dvdcol'])
    c25=fits.Column(name="nbr_psf_fit_pars",format="18D",array=nbr_info['nbr_psf_fit_pars'])

    cols=fits.ColDefs([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.header.set('EXTNAME','nbrs_data')
    return tbhdu

def save_copy_mof_file(moffile,new_ext,newmoffile):
    '''code to replace neighbor extension and write out new mof file'''

    mof=fits.open(moffile)
    mof['nbrs_data']=new_ext

    mof.writeto(newmoffile,overwrite=True)


def main(params):
    
    # check the input parameters
    check_input(params)

    # define full filepaths and names
    medsfile = os.path.join(params.dir,params.medsfile)
    moffile  = os.path.join(params.dir,params.moffile)
    mofnamebase,_ =params.moffile.split(".")
    newmoffile = os.path.join(params.dir,mofnamebase+"_newnbrs.fits")

    # generate the lists of main galaxy id and neighbor galaxy id
    main_gal_list,nbr_gal_list,totalgals = gen_neighbor_list(medsfile)

    # determine start and end points for different processes based on
    # total number of galaxies and num processors specified
    ngalsperproc=totalgals//params.num_proc
    start=[]
    end=[]
    for nc in xrange(params.num_proc):
        start.append(nc*ngalsperproc)
        end.append((nc+1)*ngalsperproc-1)
        if nc == params.num_proc-1:
            end[nc]=totalgals-1

    # setup dictionary with info for making neighbor dictionaries
    info={'MOFFILE':moffile,
          'main_gal_list':main_gal_list,
          'nbr_gal_list':nbr_gal_list,
          'start':start[0],
          'end':end[0]}

    # copy into list of dictionaries for number of processes
    allinfo=[copy.deepcopy(info) for _ in xrange(params.num_proc)]
    
    # replace start and end points for each info dictionary
    for nc in xrange(params.num_proc):
        allinfo[nc]['start']=start[nc]
        allinfo[nc]['end']=end[nc]

    # setup pool and send to processors
    pool=multiprocessing.Pool(processes=params.num_proc)
    all_mof_nbr_info=pool.map(gen_neighbor_info,allinfo)

    # combine dictionaries from different processors
    mof_nbr_info=combine_dictionaries(all_mof_nbr_info,params.num_proc)

    # create a new extension
    new_ext=create_new_extension(mof_nbr_info)

    # save a copy of mof file with new neighbor extension
    save_copy_mof_file(moffile,new_ext,newmoffile)

    return

if __name__ == '__main__':
    params = parse_input()

    aa=time.clock()

    # run program to produced target/template galaxies and save their moments in a fits file
    main(params)

    bb=time.clock()
    print("run time %s" %(bb-aa))


