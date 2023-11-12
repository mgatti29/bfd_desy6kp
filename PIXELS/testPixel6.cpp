// Get covariance matrix of moments of shifted noise.

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
  "testPixel6: Measure covariance matrix of shifted noise.";

const int UseMoments = USE_ALL_MOMENTS;
typedef MomentIndices<UseMoments> MI;

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

int main(int argc,
	 char *argv[])
{
  double flux = 0.;
  double sigmaG = 1.5;
  double sigmaW = 2.;
  double sigmaPSF = 1.5;
  double noise=1.;

  long nTarget = atol(argv[1]);
  double dx = atof(argv[2]);

  try {
    ran::UniformDeviate ud;
    const GaussianWeight kw(sigmaW);
    sbp::SBGaussian psf0(sigmaPSF);
    SbPsf psf(psf0);

    
    Ellipse ell(Shear(0.,0.), log(sigmaG), Position<double>(0.,0.));
    // Get the galaxy covariance matrix 
    MomentCovariance<UseMoments> mcov;
    {
      TemplateGalaxy<UseMoments>* gg = ellipse2galaxy(kw, psf, flux, ell, noise, ud, true);
      mcov = gg->getCov();
      delete gg;
    }
    cout << "Calculated:\n" << mcov << endl;

    DVector msum(6,0.);
    DMatrix covsum(6,6,0.);
    long nsum=0;

    for (long itarget=0; itarget<nTarget; itarget++) {
      // Create the sheared, noisy galaxy
      TemplateGalaxy<UseMoments>* gg = ellipse2galaxy(kw, psf, flux, ell, noise, ud, true);
      Galaxy<UseMoments>* shifted = gg->getShifted(dx,0.);
      MI::MVector m = shifted->getMoments();
      delete gg;
      delete shifted;
      DVector mm(6);
      /**/cout << m[1] << " " << m[2] << " " << m[0] << endl;
      for (int i=0; i<6; i++) mm[i] = m[i];
      msum += mm;
      covsum += mm^mm;
      nsum++;
    }

    msum /= (double) nsum;
    covsum /= (double) nsum;
    covsum -= msum ^ msum;

    cout << "Mean: " << msum << endl;
    cout << "Observed covariance:\n" << covsum << endl;

  } catch (std::runtime_error& m) {
    quit(m,1);
  }

  exit(0);
}
