import numpy as np
import astropy.io.fits as fits
import pdb

class gvals(object):
    def __init__(self):
        self.g1=None
        self.g2=None
        self.eg1=None
        self.eg2=None

class mcvals(object):
    def __init__(self,separate=False):
        if separate:
            self.m1=None
            self.m2=None
            self.c1=None
            self.c2=None
            self.em1=None
            self.em2=None
            self.ec1=None
            self.ec2=None
        else:
            self.m=None
            self.c=None
            self.em=None
            self.ec=None

class read_bfd_moment_file(object):
    
    def __init__(self,file,target=True,pqr=True):
        
        x=fits.open(file)
        print(target,pqr)
        self.n = np.size(x[1].data['id'])

        self.MF = x[1].data['moments'][:,0]
        self.MR = x[1].data['moments'][:,1]
        self.M1 = x[1].data['moments'][:,2]
        self.M2 = x[1].data['moments'][:,3]
        self.MC = x[1].data['moments'][:,4]

        if (target == True):

            self.nlost = x[0].header['nlost']
            self.covXX = x[0].header['covmxmx']
            self.covXY = x[0].header['covmxmy']
            self.covYY = x[0].header['covmymy']
            self.covFF = x[0].data[0][0]
            self.covRR = x[0].data[1][1]
            self.cov11 = x[0].data[2][2]
            self.cov22 = x[0].data[3][3]
            self.covCC = x[0].data[4][4]
            if (pqr == True):
                self.P   = x[1].data['PQR'][:,0]
                self.Q1  = x[1].data['PQR'][:,1]
                self.Q2  = x[1].data['PQR'][:,2]
                self.R11 = x[1].data['PQR'][:,3]
                self.R12 = x[1].data['PQR'][:,4]
                self.R22 = x[1].data['PQR'][:,5]
            else:
                self.P = None
                self.Q1 = None
                self.Q2 = None
                self.R11 = None
                self.R12 = None
                self.R22 = None

        else:

            self.MX = x[1].data['moments'][:,5]
            self.MY = x[1].data['moments'][:,6]

            self.MF_dg1 = x[1].data['moments_dg1'][:,0]
            self.MR_dg1 = x[1].data['moments_dg1'][:,1]
            self.M1_dg1 = x[1].data['moments_dg1'][:,2]
            self.M2_dg1 = x[1].data['moments_dg1'][:,3]
            self.MC_dg1 = x[1].data['moments_dg1'][:,4]
            self.MX_dg1 = x[1].data['moments_dg1'][:,5]
            self.MY_dg1 = x[1].data['moments_dg1'][:,6]

            self.MF_dg2 = x[1].data['moments_dg2'][:,0]
            self.MR_dg2 = x[1].data['moments_dg2'][:,1]
            self.M1_dg2 = x[1].data['moments_dg2'][:,2]
            self.M2_dg2 = x[1].data['moments_dg2'][:,3]
            self.MC_dg2 = x[1].data['moments_dg2'][:,4]
            self.MX_dg2 = x[1].data['moments_dg2'][:,5]
            self.MY_dg2 = x[1].data['moments_dg2'][:,6]

            self.MF_dmu = x[1].data['moments_dmu'][:,0]
            self.MR_dmu = x[1].data['moments_dmu'][:,1]
            self.M1_dmu = x[1].data['moments_dmu'][:,2]
            self.M2_dmu = x[1].data['moments_dmu'][:,3]
            self.MC_dmu = x[1].data['moments_dmu'][:,4]
            self.MX_dmu = x[1].data['moments_dmu'][:,5]
            self.MY_dmu = x[1].data['moments_dmu'][:,6]

            self.MF_dg1dg1 = x[1].data['moments_dg1_dg1'][:,0]
            self.MR_dg1dg1 = x[1].data['moments_dg1_dg1'][:,1]
            self.M1_dg1dg1 = x[1].data['moments_dg1_dg1'][:,2]
            self.M2_dg1dg1 = x[1].data['moments_dg1_dg1'][:,3]
            self.MC_dg1dg1 = x[1].data['moments_dg1_dg1'][:,4]
            self.MX_dg1dg1 = x[1].data['moments_dg1_dg1'][:,5]
            self.MY_dg1dg1 = x[1].data['moments_dg1_dg1'][:,6]

            self.MF_dg1dg2 = x[1].data['moments_dg1_dg2'][:,0]
            self.MR_dg1dg2 = x[1].data['moments_dg1_dg2'][:,1]
            self.M1_dg1dg2 = x[1].data['moments_dg1_dg2'][:,2]
            self.M2_dg1dg2 = x[1].data['moments_dg1_dg2'][:,3]
            self.MC_dg1dg2 = x[1].data['moments_dg1_dg2'][:,4]
            self.MX_dg1dg2 = x[1].data['moments_dg1_dg2'][:,5]
            self.MY_dg1dg2 = x[1].data['moments_dg1_dg2'][:,6]

            self.MF_dg2dg2 = x[1].data['moments_dg2_dg2'][:,0]
            self.MR_dg2dg2 = x[1].data['moments_dg2_dg2'][:,1]
            self.M1_dg2dg2 = x[1].data['moments_dg2_dg2'][:,2]
            self.M2_dg2dg2 = x[1].data['moments_dg2_dg2'][:,3]
            self.MC_dg2dg2 = x[1].data['moments_dg2_dg2'][:,4]
            self.MX_dg2dg2 = x[1].data['moments_dg2_dg2'][:,5]
            self.MY_dg2dg2 = x[1].data['moments_dg2_dg2'][:,6]

            self.MF_dg1dmu = x[1].data['moments_dmu_dg1'][:,0]
            self.MR_dg1dmu = x[1].data['moments_dmu_dg1'][:,1]
            self.M1_dg1dmu = x[1].data['moments_dmu_dg1'][:,2]
            self.M2_dg1dmu = x[1].data['moments_dmu_dg1'][:,3]
            self.MC_dg1dmu = x[1].data['moments_dmu_dg1'][:,4]
            self.MX_dg1dmu = x[1].data['moments_dmu_dg1'][:,5]
            self.MY_dg1dmu = x[1].data['moments_dmu_dg1'][:,6]

            self.MF_dg2dmu = x[1].data['moments_dmu_dg2'][:,0]
            self.MR_dg2dmu = x[1].data['moments_dmu_dg2'][:,1]
            self.M1_dg2dmu = x[1].data['moments_dmu_dg2'][:,2]
            self.M2_dg2dmu = x[1].data['moments_dmu_dg2'][:,3]
            self.MC_dg2dmu = x[1].data['moments_dmu_dg2'][:,4]
            self.MX_dg2dmu = x[1].data['moments_dmu_dg2'][:,5]
            self.MY_dg2dmu = x[1].data['moments_dmu_dg2'][:,6]

            self.MF_dmudmu = x[1].data['moments_dmu_dmu'][:,0]
            self.MR_dmudmu = x[1].data['moments_dmu_dmu'][:,1]
            self.M1_dmudmu = x[1].data['moments_dmu_dmu'][:,2]
            self.M2_dmudmu = x[1].data['moments_dmu_dmu'][:,3]
            self.MC_dmudmu = x[1].data['moments_dmu_dmu'][:,4]
            self.MX_dmudmu = x[1].data['moments_dmu_dmu'][:,5]
            self.MY_dmudmu = x[1].data['moments_dmu_dmu'][:,6]


    def get_g(self):
        gg=gvals()
        if self.P is None:
            raise Exception("Must have PQRs")
        
        gd=np.where(self.P > 0.0)
        sum1a=np.sum(self.Q1[gd]/self.P[gd])
        sum1b=np.sum(self.Q2[gd]/self.P[gd])
        sum2a=np.sum((self.Q1[gd]*self.Q1[gd])/self.P[gd]**2 - self.R11[gd]/self.P[gd])
        sum2b=np.sum((self.Q1[gd]*self.Q2[gd])/self.P[gd]**2 - self.R12[gd]/self.P[gd])
        sum2c=np.sum((self.Q2[gd]*self.Q1[gd])/self.P[gd]**2 - self.R12[gd]/self.P[gd])
        sum2d=np.sum((self.Q2[gd]*self.Q2[gd])/self.P[gd]**2 - self.R22[gd]/self.P[gd])

        C = np.matrix([[sum2a,sum2b],[sum2c,sum2d]])
        Cinv = np.linalg.inv(C)
        Q_P = np.matrix([[sum1a],[sum1b]])

        gg.g1 = (Cinv*Q_P)[0,0]
        gg.g2 = (Cinv*Q_P)[1,0]
        gg.eg1 = np.sqrt(Cinv[0,0])
        gg.eg2 = np.sqrt(Cinv[1,1])
        return gg

    def get_mc(self,g1,g2):
        mc=mcvals()
        if self.P is None:
            raise Exception("Must have PQRs")
        if len(g1) != len(self.P):
            raise Exception("length of input g must be same as PQRs")
        
        gd=np.where(self.P > 0.0)
        term1a=self.Q1[gd]/self.P[gd]
        term1b=self.Q2[gd]/self.P[gd]
        term2a=(self.Q1[gd]*self.Q1[gd])/self.P[gd]**2 - self.R11[gd]/self.P[gd]
        term2b=(self.Q1[gd]*self.Q2[gd])/self.P[gd]**2 - self.R12[gd]/self.P[gd]
        term2c=(self.Q2[gd]*self.Q1[gd])/self.P[gd]**2 - self.R12[gd]/self.P[gd]
        term2d=(self.Q2[gd]*self.Q2[gd])/self.P[gd]**2 - self.R22[gd]/self.P[gd]
        g1=g1[gd]
        g2=g2[gd]

        sum1= np.sum(g1*term1a+g2*term1b)
        sum2= np.sum(term1a + term1b)
        sum3= np.sum(g1*(g1*term2a+g2*term2c)+g2*(g1*term2b+g2*term2d))
        sum4= np.sum(g1*term2a+g2*term2c + g1*term2b+g2*term2d)
        sum5= np.sum(g1*(term2a + term2c) + g2*(term2b+term2d))
        sum6= np.sum(term2a+term2b+term2c+term2d)

        C = np.matrix([[sum3,sum4],[sum5,sum6]])
        Cinv = np.linalg.inv(C)
        Q_P = np.matrix([[sum1],[sum2]])

        mc.m = (Cinv*Q_P)[0,0]-1.0
        mc.c = (Cinv*Q_P)[1,0]
        mc.em = np.sqrt(Cinv[0,0])
        mc.ec = np.sqrt(Cinv[1,1])
        return mc

    def get_mc_separate(self,g1,g2):
        mc=mcvals(separate=True)
        if self.P is None:
            raise Exception("Must have PQRs")
        if len(g1) != len(self.P):
            raise Exception("length of input g must be same as PQRs")
        
        gd=np.where(self.P > 0.0)
        term1a=self.Q1[gd]/self.P[gd]
        term1b=self.Q2[gd]/self.P[gd]
        term2a=(self.Q1[gd]*self.Q1[gd])/self.P[gd]**2 - self.R11[gd]/self.P[gd]
        term2bc=(self.Q1[gd]*self.Q2[gd])/self.P[gd]**2 - self.R12[gd]/self.P[gd]
        term2d=(self.Q2[gd]*self.Q2[gd])/self.P[gd]**2 - self.R22[gd]/self.P[gd]
        g1=g1[gd]
        g2=g2[gd]

        sumb1= np.sum(g1*term1a)
        sumb2= np.sum(term1a)
        sumb3= np.sum(g2*term1b)
        sumb4= np.sum(term1b)

        sumA11= np.sum(term2a*g1**2)
        sumA12= np.sum(term2a*g1)
        sumA13= np.sum(term2bc*g1*g2)
        sumA14= np.sum(term2bc*g1)
        sumA22= np.sum(term2a)
        sumA23= np.sum(term2bc*g2)
        sumA24= np.sum(term2bc)
        sumA33= np.sum(term2d*g2**2)
        sumA34= np.sum(term2d*g2)
        sumA44= np.sum(term2d)


        C = np.matrix([[sumA11,sumA12,sumA13,sumA14],[sumA12,sumA22,sumA23,sumA24],[sumA13,sumA23,sumA33,sumA34],[sumA14,sumA24,sumA34,sumA44]])
        Cinv = np.linalg.inv(C)
        Q_P = np.matrix([[sumb1],[sumb2],[sumb3],[sumb4]])

        mc.m1 = (Cinv*Q_P)[0,0]-1.0
        mc.c1 = (Cinv*Q_P)[1,0]
        mc.m2 = (Cinv*Q_P)[2,0]-1.0
        mc.c2 = (Cinv*Q_P)[3,0]

        mc.em1 = np.sqrt(Cinv[0,0])
        mc.ec1 = np.sqrt(Cinv[1,1])
        mc.em2 = np.sqrt(Cinv[2,2])
        mc.ec2 = np.sqrt(Cinv[3,3])
        return mc


# ??? Do we need this class? It's picked out in some merge.
class make_band_weight_file(object):
    
    def __init__(self,nbands):
        '''initialize an instance of particular object to find best weighting'''
        self.nbands=nbands
        self.moment=[]
        self.covariance=[]
        self.derivative=[]
        self.id=[]

        prihdr=fits.Header()
        self.prihdu=fits.PrimaryHDU(header=prihdr)

        return

    def add_gal(self,mc,id=0,usemoment=2):
        '''given a moment calculator, get moments, covariance, and derivatives
        usemoment=0 is flux moment
        usemoment=1 is R moment
        usemoment=2 is M1 (+) moment
        usemoment=3 is M2 (x) moment
        usemoment=4 is MC moment
        default is 2
        '''
        self.id.append(id)
        mc.recenter()
        m,mb=mc.get_moment(0,0,returnbands=True)
        ce,co,cbe,cbo=mc.get_covariance(returnbands=True)
        t,tb=mc.get_template(0,0,returnbands=True)
        mom=[]
        cov=[]
        der=[]

        for b in mb:
            mom.append(b.even[usemoment])
        for i in xrange(self.nbands):
            cov.append(cbe[usemoment,usemoment,i])

        for b in tb:
            der.append(b.get_dg1().even[usemoment])

        self.moment.append(mom)
        self.covariance.append(cov)
        self.derivative.append(der)

        return

    def save_file(self,filename):
        
        formattype=str(self.nbands)+"E"

        col1=fits.Column(name="id",format="K",array=self.id)
        col2=fits.Column(name="moments",format=formattype,array=self.moment)
        col3=fits.Column(name="covs",format=formattype,array=self.covariance)
        col4=fits.Column(name="derivs",format=formattype,array=self.derivative)
        cols=fits.ColDefs([col1,col2,col3,col4])

        tbhdu=fits.BinTableHDU.from_columns(cols)
        
        thdulist=fits.HDUList([self.prihdu,tbhdu])
        thdulist.writeto(filename,overwrite=True)
        return
