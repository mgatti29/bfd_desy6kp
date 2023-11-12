// Compare distributions of centroid and moments for fixed noise vs shifting noise.
// Differential moments will be recorded w.r.t. the true underlying galaxy.

#include "Pqr.h"
#include "Distributions.h"
#include "Shear.h"
#include "Moments.h"
#include "Galaxy.h"
#include "Random.h"
#include "StringStuff.h"
#include "Prior.h"
#include "SbPsf.h"
#include "FitsImage.h"
#include "FitsTable.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace bfd;
const string usage = 
  "testPixel7: Record distribution of centroid positions and moments for 2 different noise types";

const int UseMoments = USE_ALL_MOMENTS;
typedef MomentIndices<UseMoments> MI;

TemplateGalaxy<UseMoments>*
newtonShift(const TemplateGalaxy<UseMoments>& gin, 
	    double& cx, double& cy,
	    const MomentCovariance<UseMoments>& cov,
	    int iterations=4) {
  // Do fixed number of Newton iteration to bring the xy moments near zero:
  cx = cy = 0.;
  const TemplateGalaxy<UseMoments>* gbase=&gin;
  TemplateGalaxy<UseMoments>* shifted=0;
  for (int iter=0; iter<iterations; iter++) {
    DVector2 dm;
    dm[0] = gbase->getMoments()[MI::CX];
    dm[1] = gbase->getMoments()[MI::CY];
    DMatrix22 dmdx;
    dmdx(0,0) = gbase->dMdx()[MI::CX];
    dmdx(0,1) = gbase->dMdy()[MI::CX];
    dmdx(1,0) = gbase->dMdx()[MI::CY];
    dmdx(1,1) = gbase->dMdy()[MI::CY];
    dm /= dmdx;
    cx += dm[0];
    cy += dm[1];
    if (shifted) delete shifted;
    shifted = gin.getShifted(-cx,-cy);
    gbase = shifted;
    /**     dm[0] = gbase->getMoments()[MI::CX];
    dm[1] = gbase->getMoments()[MI::CY];
    cerr << iter 
	<< " ctr " << cx << " " << cy
	<< " moments " << dm 
	<< " sigma " << dm / (double) sqrt(cov(MI::CX,MI::CX)) 
	<< endl; 
	/**/
  }
  MI::MVector m = shifted->getMoments();
  double chisq = m[MI::CX]*m[MI::CX] / cov(MI::CX,MI::CX) +
    m[MI::CY]*m[MI::CY] / cov(MI::CY,MI::CY);
  if (chisq > 1e-10) {
    cerr << "Warning: newton chisq " << chisq << endl;
  }
  return shifted;
}

// Class for galaxy with fixed moment noise
class FNGalaxy: public TemplateGalaxy<UseMoments> {
public:
  FNGalaxy(const KWeight& kw_,
	   const sbp::SBProfile& sbg,
	   const Psf& psf,
	   ran::UniformDeviate& ud,
	   double noise=1.): TemplateGalaxy<UseMoments>(kw_) {
    // Base is a noiseless version
    base = SbGalaxy<UseMoments>(kw, sbg, psf, ud, noise, false);
    // Get a noisy version of same galaxy, save the noise
    noisy = SbGalaxy<UseMoments>(kw, sbg, psf, ud, noise, true);
    mnoise = noisy->getMoments() - base->getMoments();
  }
  FNGalaxy(const KWeight& kw_,
	   TemplateGalaxy<UseMoments>* base_,
	   TemplateGalaxy<UseMoments>* noisy_,
	   MI::MVector mnoise_): TemplateGalaxy<UseMoments>(kw_),
    base(base_), noisy(noisy_), mnoise(mnoise_) {}
  FNGalaxy(const FNGalaxy& rhs): TemplateGalaxy<UseMoments>(rhs.kw),
    base(rhs.base->duplicate()), noisy(rhs.noisy->duplicate()), mnoise(rhs.mnoise) {}
  virtual FNGalaxy* duplicate() const {return new FNGalaxy(*this);}
  virtual ~FNGalaxy() {delete base; delete noisy;}
  
  const TemplateGalaxy<UseMoments>* getBase() const {return base;}
  const TemplateGalaxy<UseMoments>* getNoisy() const {return noisy;}
  
  virtual Moments<UseMoments> getMoments() const {
    MI::MVector out =base->getMoments();
    out += mnoise;
    return out;
  }
  virtual TemplateGalaxy<UseMoments>* getShifted(double dx, double dy) const {
    return new FNGalaxy(kw, base->getShifted(dx,dy),noisy->getShifted(dx,dy), mnoise);
  }
  virtual MomentCovariance<UseMoments> getCov() const {return base->getCov();}
  virtual Moments<UseMoments> dMdx() const {return base->dMdx();}
  virtual Moments<UseMoments> dMdy() const {return base->dMdy();}
  virtual MomentDecomposition<UseMoments> getDecomposition() const {return base->getDecomposition();}
private:
  TemplateGalaxy<UseMoments>* base;
  TemplateGalaxy<UseMoments>* noisy;
  MI::MVector mnoise;
};


// Class that produces Galaxies of chosen qualities
class GalaxyMaker{
public:
  GalaxyMaker(const KWeight& kw_, const Psf& psf_,
	      double sn_,
	      double noise_, 
	      double sigmaG_,
	      Shear g_,
	      ran::UniformDeviate& ud_): kw(kw_), psf(psf_),
					 ud(ud_),
					 noise(noise_),
					 ell(g_, log(sigmaG_), Position<double>(0.,0.)),
					 sbgal(0) {
    // Create SBProfile for the desired galaxy, choose flux to give selected S/N.
    flux = 1.;
    sbp::SBGaussian src(1.,1.);
    sbgal = src.distort(ell);
    sbgal->setFlux(1.);
    TemplateGalaxy<UseMoments>* test = SbGalaxy<UseMoments>(kw, *sbgal, psf, ud, noise, false);
    flux *= sn_*sqrt(test->getCov()(MI::FLUX,MI::FLUX)) / test->getMoments()[MI::FLUX];
    sbgal->setFlux(flux);
    delete test;
  }
  ~GalaxyMaker() {delete sbgal;}

  TemplateGalaxy<UseMoments>* getGal(bool perfect=false) {
    // Get a version of the galaxy.
    // perfect=True gives noiseless analytic Gaussian.
    // perfect=False gives dual-noise SbGalaxy.

    if (perfect) {
      double e =  ell.getS().getE();
      double beta = ell.getS().getBeta();
      double sigma = exp(ell.getMu()) * pow(1-e*e, -0.25);
      return new GaussianGalaxy<UseMoments>(kw, flux, sigma, e, beta, 0., 0., 0.);
    } else {
      return new FNGalaxy(kw, *sbgal, psf, ud, noise);
    }
  }

private:
  const KWeight& kw;
  const Psf& psf;
  ran::UniformDeviate& ud;
  double noise;
  Ellipse ell;
  double flux;
  sbp::SBProfile* sbgal;
};

int main(int argc,
	 char *argv[])
{
  double sigmaG = 1.5;
  double sigmaW = 2.;
  double sigmaPSF = 1.5;
  double noise=1.;

  long ntrials = atol(argv[1]);
  double sn= atof(argv[2]);
  double g1 = 0.;
  double g2 = 0.;
  Shear g;
  g.setG1G2(g1,g2);
  string fname = "mom12.fits";
  if (argc>3) fname = argv[3];

  try {
    ran::UniformDeviate ud;

    const GaussianWeight kw(sigmaW);
    sbp::SBGaussian psf0(sigmaPSF);
    SbPsf psf(psf0);

    // Make vectors holding output centroids and moments
    vector<float> x1;
    vector<float> y1;
    vector<float> x2;
    vector<float> y2;
    vector<float> p1;
    vector<float> p2;
    vector<vector<float> > mom1;
    vector<vector<float> > mom2;

    // Make the underlying galaxy
    GalaxyMaker factory(kw, psf, sn, noise, sigmaG, g, ud);
    
    // Get a perfect galaxy:
    TemplateGalaxy<UseMoments>* perfect = factory.getGal(true);

    // And one real one to get a covariance matrix
    MomentCovariance<UseMoments> mcov;
    {
      TemplateGalaxy<UseMoments>* tmp = factory.getGal();
      mcov = tmp->getCov();
      delete tmp;
    }
    cout << "Moments: " << perfect->getMoments() << endl;
    cout << "Covariance matrix: " << setw(6) << mcov << endl;
    // Save inverse of xy moment covariance:
    double det = mcov(MI::CX,MI::CX) * mcov(MI::CY,MI::CY) -
      mcov(MI::CX,MI::CY) * mcov(MI::CX,MI::CY);
    double invcxx = mcov(MI::CY,MI::CY) / det;
    double invcyy = mcov(MI::CX,MI::CX) / det;
    double invcxy = -mcov(MI::CX,MI::CY) / det;

    for (long itrial=0; itrial<ntrials; itrial++) {
      if ((itrial+1)%50000==0) cerr << fname << " trial " << itrial << endl;
      // Make a pair of objects with fixed/variable noise
      FNGalaxy* fng = dynamic_cast<FNGalaxy*> (factory.getGal());

      // Newton shift fixed noise
      double cx, cy;
      TemplateGalaxy<UseMoments>* shifted = newtonShift(*fng, cx, cy, mcov, 4);
      MI::MVector m = shifted->getMoments();
      delete shifted;
      
      // Make noiseless object with given shift
      shifted = perfect->getShifted(cx,cy);
      MI::MVector mp = shifted->getMoments();
      vector<float> vm(6);
      for (int i=0; i<6; i++) vm[i] = m[i] - mp[i];

      // Calculate probability density at this centroid
      MomentDerivs<UseMoments> md = shifted->getDerivs();
      delete shifted;
      double p = exp(-0.5*(mp[MI::CX]*mp[MI::CX]*invcxx +
			     mp[MI::CY]*mp[MI::CY]*invcyy +
			      2.*mp[MI::CX]*mp[MI::CY]*invcxy));
      p *= md.jacobianPqr()[Pqr::P];
      
      // Save information
      x1.push_back(cx);
      y1.push_back(cy);
      mom1.push_back(vm);
      p1.push_back(p);

      // Repeat for the variable-noise object
      // Newton shift fixed noise
      cx = cy = 0.;
      shifted = newtonShift(*(fng->getNoisy()), cx, cy, mcov, 4);
      m = shifted->getMoments();
      delete shifted;
      
      // Make noiseless object with given shift
      shifted = fng->getBase()->getShifted(cx,cy);
      mp = shifted->getMoments();
      for (int i=0; i<6; i++) vm[i] = m[i] - mp[i];

      // Calculate probability density at this centroid
      md = shifted->getDerivs();
      delete shifted;
      p = exp(-0.5*(mp[MI::CX]*mp[MI::CX]*invcxx +
		    mp[MI::CY]*mp[MI::CY]*invcyy +
		    2.*mp[MI::CX]*mp[MI::CY]*invcxy));
      p *= md.jacobianPqr()[Pqr::P];
      
      delete fng;

      // Save information
      x2.push_back(cx);
      y2.push_back(cy);
      mom2.push_back(vm);
      p2.push_back(p);
    }

    // Save data into a FITS table

    img::FTable tab;
    tab.addColumn(x1, "X1");
    tab.addColumn(y1, "Y1");
    tab.addColumn(x2, "X2");
    tab.addColumn(y2, "Y2");
    tab.addColumn(p1, "P1");
    tab.addColumn(p2, "P2");
    tab.addColumn(mom1, "MOM1");
    tab.addColumn(mom2, "MOM2");
      
    {
      FITS::FitsTable ft(fname,FITS::ReadWrite + FITS::Create, 1);
      ft.copy(tab);
    }
      
  } catch (std::runtime_error& m) {
    quit(m,1);
  }

  exit(0);
}
