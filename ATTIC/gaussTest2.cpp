// Simple Gaussian test program using BFD code.
#include "Pqr.h"
#include "Distributions.h"
#include "Shear.h"
#include "Moments.h"
#include "Galaxy.h"
#include "Random.h"
#include "StringStuff.h"
#include "Prior.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace bfd;
const string usage = 
  "gaussTest2: estimate shear with BFD method using Gaussian galaxies and weight function.\n"
  "            Moments are calculated analytically.\n"
  "            Galaxies have known centroids and uniform distributions of flux & size.\n"
  "            Moments in use determined at compile time.\n"
  "            Galaxies and templates are rotated near E2=0.\n"
  "            Brute-force integration over the template set.\n"
  "            Weight size is set to equal galaxy size.\n"
  "Usage: gaussTest2 <sigma_e> <g1> <g2> <N_target> <N_template> [midSn=15] [widthSn=10] \n"
  "                  [midSigmaG=1.5] [widthSigmaG=1.0]\n"
  "       <sigma_e> is sigma of intrinsic ellipticity distribution, exp(-e^2/2 sigma^2)(1-e^2)^2\n"
  "       <g1> <g2> are applied shear\n"
  "       <N_target> is number of galaxies ""observed""\n"
  "       <N_template> is number of noiseless template galaxies drawn from distribution\n"
  "       [midSn], ...are the middle and width of the (uniform) distributions of galaxy S/N and size\n"
  "Output: The summed P,Q,R of the targets and the estimated shear & uncertainty";


const int UseMoments = USE_E_MOMENT;
typedef MomentIndices<UseMoments> MI;

GaussianGalaxy<UseMoments> 
ellipse2galaxy(const KWeight& kw, double flux, const Ellipse& el, double noise) {
  double e =  el.getS().getE();
  double beta = el.getS().getBeta();
  Position<double> c = el.getX0();
  double sigma = exp(el.getMu()) * pow(1-e*e, -0.25);
  return GaussianGalaxy<UseMoments>(kw, flux, sigma, e, beta, c.x, c.y, noise);
}

// Make new class for making galaxy 'factories'
class galaxyDistribution{

  public:
  galaxyDistribution (const KWeight& kw_, 
		      const double sigmaE_, 
		      const double midSn_, 
		      const double widthSn_,
		      const double noise_, 
		      const double midSigmaG_,
		      const double widthSigmaG_, 
		      ran::UniformDeviate& ud_ ): kw(kw_), 
						  noise(noise_),
						  midSigmaG(midSigmaG_),
						  widthSigmaG(widthSigmaG_),
						  midSn(midSn_),
						  widthSn(widthSn_),
						  ud(ud_), 
						  eDist(sigmaE_, ud_) {}

  // Methods:
  // get Galaxy method (default no shear.  For shear, put shear as arg)
  GaussianGalaxy<UseMoments> getGal(Shear g=Shear(0.,0.)){
    double e1, e2, sn, sigmaG;
#ifdef _OPENMP
#pragma omp critical(random)
#endif
    {    
      eDist.sample(e1, e2);
      sn = (ud()-0.5) * widthSn + midSn; // Each drawn galaxy has its own sn
      sigmaG = (ud()-0.5) * widthSigmaG + midSigmaG;
    }
    Ellipse distort(g, 0., Position<double>(0.,0.));  // The distortion applied to each target
    Ellipse ell(Shear(e1,e2), log(sigmaG), Position<double>(0.,0.));
    double flux = sn * sqrt(4*PI*sigmaG*sigmaG*noise);
    GaussianGalaxy<UseMoments> gg = ellipse2galaxy(kw, flux, distort + ell, noise);
    
    return  gg;
  } 

  private:
  const KWeight& kw;
  double noise;
  double sigmaG;
  double midSigmaG;
  double widthSigmaG;
  double midSn;
  double widthSn;
  ran::UniformDeviate& ud;
  BobsEDist eDist;	// The parent ellipticity distribution
};

  // galaxyDistribution constructor 

int main(int argc,
	 char *argv[])
{
  if (argc < 6 || argc >10) {
    cerr << usage << endl;
    exit(1);
  }

  //  New command line args:
  //  1 sigmaE 0.2
  //  2 G1G2[1] 0.01
  //  3 G1G2[2] 0.0
  //  4 nTarget 10 
  //  5 nTemplate 10 
  //  6 midSn 
  //  7 widthSN 
  //  8 midSigmaG 
  //  9 widthSigmaG 

  const double sigmaW = 1.;
  double sigmaE = atof(argv[1]);
  Shear g;
  g.setG1G2(atof(argv[2]), atof(argv[3]));
  long nTarget = atol(argv[4]);
  long nTemplate = atol(argv[5]);
  double midSn = argc > 6 ? atof(argv[6]) : 15.;
  double widthSn = argc > 7 ? atof(argv[7]) : 10.;
  double midSigmaG = argc > 8 ? atof(argv[8]) : 1.5;
  double widthSigmaG = argc > 9 ? atof(argv[9]) : 1.;

  ran::UniformDeviate ud;
  ran::GaussianDeviate gd(ud); //e.g. float f = ud() to get a single number

  const double noise=1.;	// White noise level

  try {
    cout << "#" << stringstuff::taggedCommandLine(argc,argv) << endl;
    cout << "# Number of moments: " << MI::N << endl;

    const GaussianWeight kw(sigmaW);

    // Make galaxy distribution: same distribution for template & target
    galaxyDistribution templateDistribution( kw, sigmaE, midSn, widthSn, noise, 
					     midSigmaG, widthSigmaG, ud);

    // Create one galaxy to get the noise covariance matrix and make noise generator
    MomentCovariance<UseMoments> mcov;
    
    mcov = templateDistribution.getGal().getCov();

    // Create prior with very loose cutoffs, and signalling centers of galaxies are known.
    double fluxMin = 0.;
    double fluxMax = 200.;
    if (MI::UseFlux) {
      fluxMax = sqrt(mcov(MI::FLUX,MI::FLUX)) * (midSn + 0.5*widthSn + 10.);
    }
    SampledPrior<UseMoments> prior(fluxMin, fluxMax, true);

    prior.setNominalCovariance(mcov);
    prior.setInvariantCovariance(true);

    /**/cerr << "Ready for templates" << endl;
    // Draw template galaxies
    for (long i = 0; i<nTemplate; i++) {
      GaussianGalaxy<UseMoments> gg = templateDistribution.getGal();
      prior.addTemplate( gg, ud, 1., 1., false);
    }

    /**/cerr << "Done templates" << endl;

    tmv::SymMatrix<typename MI::Type> cov(MI::N);
    for (int i=0; i<cov.size(); i++)
      for (int j=0; j<=i; j++)
	cov(i,j) = mcov(i,j);
    MultiGauss<typename MI::Type> momentNoise(cov);
  
    // Prepare the prior and begin measuring galaxies
    prior.prepare();

    /**/cerr << "Done prepare" << endl;
    Pqr accumulator;

    const long chunk = MIN(100L, nTarget);	//Number of galaxies to create per for loop
    long nLoops = nTarget / chunk;

#ifdef _OPENMP
#pragma omp parallel
#endif

    {

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
      for (long i=0; i<nLoops; i++) {
	Pqr subAccum;
	for (long j=0; j<chunk; j++) {
	  double e1, e2;
	  MI::MVector m;
	  MI::MVector addnoise;
#ifdef _OPENMP
#pragma omp critical(random)
#endif
	  {
	    // Must block other threads in this block until we make parallel random-number generators
	    //eDist.sample(e1, e2);
	    addnoise = momentNoise.sample(gd);
	  }

	  // Get the noiseless moments and add noise
          GaussianGalaxy<UseMoments> gg = templateDistribution.getGal(g);
	  Moments<UseMoments> mobs(gg.getMoments() + addnoise);

	  // Calculate the Pqr for this galaxy and add it in:
	  subAccum += prior.getPqr(mobs, mcov).neglog();
	}
#ifdef _OPENMP
#pragma omp critical(accumulate)
#endif
	{
	accumulator += subAccum;
	}
      }
    }
    // Outputs:
    cout << "PQR: " << accumulator[Pqr::P] 
	 << " " << accumulator[Pqr::DG1]
	 << " " << accumulator[Pqr::DG2]
	 << " " << accumulator[Pqr::D2G1G1]
	 << " " << accumulator[Pqr::D2G2G2]
	 << " " << accumulator[Pqr::D2G1G2]
	 << endl;
    Pqr::GVector gMean;
    Pqr::GMatrix gCov;
    accumulator.getG(gMean, gCov);
    cout << "g1: " << gMean[Pqr::G1] << " +- " << sqrt(gCov(Pqr::G1,Pqr::G1))
	 << " g2: " << gMean[Pqr::G2] << " +- " << sqrt(gCov(Pqr::G2,Pqr::G2))
	 << " r: " << gCov(Pqr::G1,Pqr::G2) / sqrt(gCov(Pqr::G2,Pqr::G2)*gCov(Pqr::G1,Pqr::G1))
	 << endl;

  } catch (std::runtime_error& m) {
    quit(m,1);
  }

  exit(0);
}

       
