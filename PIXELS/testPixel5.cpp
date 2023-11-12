// Test distributions of positions that null the centroid moments

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

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace bfd;
const string usage = 
  "testPixel5: Measure distribution of centroid positions";

const int UseMoments = USE_ALL_MOMENTS;
typedef MomentIndices<UseMoments> MI;

Galaxy<UseMoments>*
newtonShift(const Galaxy<UseMoments>& gin, 
	    double& cx, double& cy,
	    const MomentCovariance<UseMoments>& cov,
	    int iterations=4) {
  // Do fixed number of Newton iteration to bring the xy moments near zero:
  cx = cy = 0.;
  const Galaxy<UseMoments>* gbase=&gin;
  Galaxy<UseMoments>* shifted=0;
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
  return shifted;
}


TemplateGalaxy<UseMoments>*
ellipse2galaxy(const KWeight& kw, const Psf& psf, double flux, const Ellipse& el, double noise,
	       ran::UniformDeviate& ud, bool addNoise=true) {
  sbp::SBGaussian src(1.,1.);
  sbp::SBProfile* src2 = src.distort(el);
  src2->setFlux(flux);
  TemplateGalaxy<UseMoments>* out = SbGalaxy<UseMoments>(kw, *src2, psf, ud, noise, addNoise);
  delete src2;
  return out;
}

// Class that produces Galaxies from specified distribution
class galaxyDistribution{
public:
  galaxyDistribution(const KWeight& kw_, const Psf& psf_,
		     double sn_,
		     double noise_, 
		     double sigmaG_,
		     ran::UniformDeviate& ud_): kw(kw_), psf(psf_),
						ud(ud_),
						sn(sn_),
						noise(noise_),
						sigmaG(sigmaG_) {
    // Determine flux scaling to S/N for average size, circular galaxy
    Ellipse ell(Shear(0.,0.), log(sigmaG), Position<double>(0.,0.));
    double flux = 1.;
    TemplateGalaxy<UseMoments>* gg = ellipse2galaxy(kw, psf, flux, ell, noise, ud, false);
    snScale = sqrt(gg->getCov()(MI::FLUX,MI::FLUX)) / gg->getMoments()[MI::FLUX];
    /**/cerr << "Setting snScale to " << snScale << endl;
    delete gg;
}

  // Galaxy sampling method
  // Default is no shear.  For shear, put shear as arg)
  // Set fixCenter = false to have galaxy centered at origin.
  TemplateGalaxy<UseMoments>* getGal(Shear g=Shear(0.,0.), bool isTemplate=false) {
    double flux;
    Position<double> ctr(0.,0.);
    TemplateGalaxy<UseMoments>* gg = 0;
#ifdef _OPENMP
#pragma omp critical(random)
#endif
    {
      // The distortion applied to each target:
      Ellipse distort(g, 0., Position<double>(0.,0.)); 
      Ellipse ell(Shear(0.,0.), log(sigmaG), ctr);
      flux = sn * snScale;
      if (isTemplate) {
	gg = new GaussianGalaxy<UseMoments>(kw, flux, sigmaG, 0., 0., 0., 0., 1.);
      } else {
	gg = ellipse2galaxy(kw, psf, flux, distort + ell, noise, ud, true);
      }
    }
    return  gg;
  } 

private:
  const KWeight& kw;
  const Psf& psf;
  ran::UniformDeviate& ud;
  double sn;
  double noise;
  double sigmaG;
  double snScale; // Conversion factor from S/N to flux.
};

int main(int argc,
	 char *argv[])
{
  double sn=20.;
  double sigmaG = 1.5;
  double sigmaW = 2.;
  double sigmaPSF = 1.5;
  double noise=1.;

  long nTarget = atol(argv[1]);
  double g1 = atof(argv[2]);
  double g2 = 0.;

  try {
    ran::UniformDeviate ud;
    ran::GaussianDeviate gd(ud);

    const GaussianWeight kw(sigmaW);
    sbp::SBGaussian psf0(sigmaPSF);
    SbPsf psf(psf0);

    // Make arrays for calculated and observed probabilities of dx,dy
    double dsig = 0.1;
    int sbox = static_cast<int> (floor(5./dsig));
    Bounds<int> b(-sbox, sbox, -sbox, sbox);
    img::Image<> prob(b,0.);
    img::Image<> count(b,0.);

    // Make galaxy distribution: same distribution for template & target
    galaxyDistribution templateDistribution( kw, psf, 
					     sn,
					     noise,
					     sigmaG,
					     ud);

    // Get the galaxy covariance matrix and build a noise generator
    MomentCovariance<UseMoments> mcov;
    {
      TemplateGalaxy<UseMoments>* gg = templateDistribution.getGal();
      mcov = gg->getCov();
      delete gg;
    }
    double det = mcov(MI::CX,MI::CX) * mcov(MI::CY,MI::CY) -
      mcov(MI::CX,MI::CY) * mcov(MI::CX,MI::CY);
    double invcxx = mcov(MI::CY,MI::CY) / det;
    double invcyy = mcov(MI::CX,MI::CX) / det;
    double invcxy = -mcov(MI::CX,MI::CY) / det;
        
    // Draw template galaxy (no noise)
    TemplateGalaxy<UseMoments>* gtmp = templateDistribution.getGal(Shear(0.,0.), true);
    // Estimate centroid error range.
    double sigx = sqrt(mcov(MI::CX,MI::CX)) / gtmp->dMdx()[MI::CX];
    /**/cerr << "sigx: " << sigx << endl;
    // Map out probability of detection vs displacement
    for (int i=-sbox; i<=sbox; i++) {
      double dy = i*dsig*sigx;
      for (int j=-sbox; j<=sbox; j++) {
	double dx = j*dsig*sigx;
	TemplateGalaxy<UseMoments>* gg = gtmp->getShifted(dx,dy);
	MI::MVector m = gg->getMoments();
	MomentDerivs<UseMoments> md = gg->getDerivs();
	Pqr jac = md.jacobianPqr();
	double pa = exp(-0.5*(m[MI::CX]*m[MI::CX]*invcxx +
			     m[MI::CY]*m[MI::CY]*invcyy +
			      2.*m[MI::CX]*m[MI::CY]*invcxy));
	double pb = jac[Pqr::P];
	double dpb = jac[Pqr::DG1];
	double ddpb = jac[Pqr::D2G1G1];
	double d1 = -( m[MI::CX] * md(MI::CX,Pqr::DG1) * invcxx +
		       m[MI::CY] * md(MI::CY,Pqr::DG1) * invcyy +
		       m[MI::CX] * md(MI::CY,Pqr::DG1) * invcxy +
		       m[MI::CY] * md(MI::CX,Pqr::DG1) * invcxy);
	double d2 = -( m[MI::CX] * md(MI::CX,Pqr::D2G1G1) * invcxx +
		       m[MI::CY] * md(MI::CY,Pqr::D2G1G1) * invcyy +
		       m[MI::CX] * md(MI::CY,Pqr::D2G1G1) * invcxy +
		       m[MI::CY] * md(MI::CX,Pqr::D2G1G1) * invcxy +
		       md(MI::CX,Pqr::DG1) * md(MI::CX,Pqr::DG1) * invcxx +
		       md(MI::CY,Pqr::DG1) * md(MI::CY,Pqr::DG1) * invcyy +
		       md(MI::CX,Pqr::DG1) * md(MI::CY,Pqr::DG1) * invcxy +
		       md(MI::CY,Pqr::DG1) * md(MI::CX,Pqr::DG1) * invcxy);
	double dpa = pa * d1;
	double ddpa = pa * (d1 * d1 + d2);
	prob(j,i) = pa*pb + g1*(pa*dpb + dpa*pb) + 0.5*g1*g1*(pa*ddpb + 2*dpa*dpb + ddpa*pb);
      }
    }
    
    /////////////////////////////////////////////////////////////////
    // Now begin measuring targets.
    /////////////////////////////////////////////////////////////////

    Shear g(0.,0.);
    g.setG1G2(g1,0.);
    
    for (long itarget=0; itarget<nTarget; itarget++) {
      // Create the sheared, noisy galaxy
      TemplateGalaxy<UseMoments>* gg = templateDistribution.getGal(g,false);

      // Recenter
      double cx,cy;
      Galaxy<UseMoments>* shifted = newtonShift(*gg, cx, cy, mcov, 4);
      delete gg;
      delete shifted;
      // Bin values of cx, cy.
      int i = static_cast<int> ( floor(cy/(sigx*dsig) + 0.5));
      int j = static_cast<int> ( floor(cx/(sigx*dsig) + 0.5));
      i = MAX(i,-sbox);
      i = MIN(i,sbox);
      j = MAX(j,-sbox);
      j = MIN(j,sbox);
      count(j,i) += 1.;
    }

    prob.shift(1,1);
    count.shift(1,1);
    img::FitsImage<>::writeToFITS("prob.fits",prob);
    img::FitsImage<>::writeToFITS("count.fits",count);

  } catch (std::runtime_error& m) {
    quit(m,1);
  }

  exit(0);
}
