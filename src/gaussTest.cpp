// BFD test code using Gaussian galaxies for moments

// #define SHIFT  // Define this macro to use centroid moments
// #define MAGNIFY   // Define this macro to use centroid moments

#include "BfdConfig.h"
#include "Pqr.h"
#include "Distributions.h"
#include "Shear.h"
#include "Moment.h"
#include "Galaxy.h"
#include "GaussianGalaxy.h"
#include "PqrAccumulator.h"
#include "Random.h"
#include "StringStuff.h"
#include "Prior.h"
#include "Pset.h"
#include "KdPrior.h"
#include "Stopwatch.h"


#ifdef _OPENMP
#include <omp.h>
#endif

using namespace bfd;

// Configure our classes
#ifdef SHIFT
const bool FIX_CENTER=false;
#else
const bool FIX_CENTER=true;
#endif

#ifdef MAGNIFY
const bool USE_MAG = true;
#else
const bool USE_MAG = false;
#endif

const bool USE_CONC = false;
const int N_COLORS=0;
const bool USE_FLOAT=true;
typedef BfdConfig<FIX_CENTER,
		  USE_CONC,
		  USE_MAG,
		  N_COLORS,
		  USE_FLOAT> BC;

// A configuration forcing double precision
typedef BfdConfig<FIX_CENTER,
		  USE_CONC,
		  USE_MAG,
		  N_COLORS,
		  false> BCD;

Pqr<BCD>& operator+=(Pqr<BCD>& lhs, const Pqr<BC>& rhs) {
  // Accumulate a float-or-double valued PQR into a double-valued one
  for (int i=0; i<BC::DSIZE; i++) lhs[i]+=rhs[i];
  return lhs;
}

typedef BC::FP FP;

// comment out this definition to revert to brute-force sampling of prior:
#define USE_TREE

const string usage = 
  "GaussTest4: Estimate shear with BFD method using Gaussian galaxies and weight function.\n"
  "            Moments are calculated analytically, no PSF, Weight function sigma defined as 1.\n"
#ifdef USE_TREE
  "            KdTree integration.\n"
#else
  "            Brute-force integration\n"
#endif
#ifdef SHIFT
  "            Centroid is variable.\n"
#else
  "            Centroid is **fixed at zero.**\n"
#endif
  "Usage: gaussTest [parameter file] [-parameter [=] value...]\n"
  "       Parameter files read in order of command line, then command-line parameters,\n"
  "       later values override earlier.\n"
  "stdin: none\n"
  "     Run with -help to get parameter list\n"
  "stdout: The summed P,Q,R of the targets and the estimated shear & uncertainty";


GaussianGalaxy<BC> 
ellipse2galaxy(double wtSigma, double flux, const Ellipse& el, double noise) {
  double e =  el.getS().getE();
  double beta = el.getS().getBeta();
  Position<double> c = el.getX0();
  double sigma = exp(el.getMu()) * pow(1-e*e, -0.25);
  return GaussianGalaxy<BC>(flux, sigma, e, beta, c.x, c.y, wtSigma, noise);
}

// Class that produces Galaxies from specified distribution
class GalaxyDistribution{
public:
  GalaxyDistribution(double wtSigma_, double sigmaE, 
		     double snMin, double snMax, double snGamma,
		     double noise_, 
		     double sigmaGMin_, const double sigmaGMax_, 
		     ran::UniformDeviate<double>& ud_,
		     const double dxy_=0.): wtSigma(wtSigma_), 
					    ud(ud_),
					    eDist(sigmaE, ud_),
					    snDist(snMin,snMax,snGamma,ud),
					    noise(noise_),
					    sigmaGMin(sigmaGMin_),
					    sigmaGMax(sigmaGMax_),
					    dxy(dxy_) {}

  // Galaxy sampling method
  // Default is no shear.  For shear, put shear as arg.
  // Centroid is randomly placed within square of side dxy.
  GaussianGalaxy<BC> getGal(Shear g=Shear(0.,0.), double mu=0.) {
    double e1, e2;
    double sn;
    double flux;
    double sigmaG;
    Position<double> ctr(0.,0.);
#ifdef _OPENMP
#pragma omp critical(random)
#endif
    {
      eDist.sample(e1, e2);
      sn = snDist.sample();
      sigmaG = sigmaGMin + ud() * (sigmaGMax - sigmaGMin);
      if (dxy>0.) {
	ctr.x = (2*ud()-1.)*dxy;
	ctr.y = (2*ud()-1.)*dxy;
      }
    }
    // The distortion applied to each target:
    flux = sn * sqrt(4*PI*sigmaG*sigmaG*noise);
    // Apply magnification and shear
    sigmaG *= (1+mu);
    flux *= (1+mu)*(1+mu);
    Ellipse distort(g, 0., Position<double>(0.,0.)); 
    Ellipse ell(Shear(e1,e2), log(sigmaG), ctr); 
    auto gg = ellipse2galaxy(wtSigma, flux, distort + ell, noise);
    
    return  gg;
  } 
private:
  const double wtSigma;
  ran::UniformDeviate<double>& ud;
  BobsEDist eDist;	// The parent ellipticity distribution
  PowerLawDistribution snDist;
  double noise;
  double sigmaGMin;
  double sigmaGMax;
  double dxy;
};

int main(int argc,
	 char *argv[])
{
  double wtSigma =1.;
  
  double sigmaE;
  double snMin;
  double snMax;
  double snGamma;
  double sigmaGMin;
  double sigmaGMax;
  double priorSigmaCutoff;
  double priorSigmaStep;
  double priorSigmaBuffer;

  double noiseFactor;
  double selectSnMin;
  double selectSnMax;

  double g1;
  double g2;
  double mu=0.;  // For non-magnifying case we will have mu=0

  long nTarget;
  long nTemplate;
  long nSample;
  int  minUniqueTemplates;

  int nThreads;
  long chunk;
  int maxLeaf;
  long seed;
  bool sampleWeights;

  Pset parameters;
  {
    const int def=PsetMember::hasDefault;
    const int low=PsetMember::hasLowerBound;
    const int up=PsetMember::hasUpperBound;
    const int lowopen = low | PsetMember::openLowerBound;
    const int upopen = up | PsetMember::openUpperBound;

    parameters.addMemberNoValue("GALAXIES:",0,
				"Specify distribution of galaxies");
    parameters.addMember("sigmaE",&sigmaE, def | lowopen,
			 "Galaxy ellipticity distribution width", 0.2, 0.);
    parameters.addMember("snMin",&snMin, def | lowopen,
			 "Minimum galaxy S/N", 5., 0.);
    parameters.addMember("snMax",&snMax, def | lowopen,
			 "Maximum galaxy S/N", 25., 0.);
    parameters.addMember("snGamma",&snGamma, def,
			 "Power law for galaxy S/N distribution", 0.);
    parameters.addMember("sigmaGMin",&sigmaGMin, def | lowopen,
			 "Minimum galaxy size", 0.5, 0.);
    parameters.addMember("sigmaGMax",&sigmaGMax, def | lowopen,
			 "Maximum galaxy size", 1.5, 0.);

    parameters.addMember("noiseFactor",&noiseFactor, def | low,
			 "Noise boost factor for kernel smoothing", 0., 0.);
    parameters.addMember("selectSnMin",&selectSnMin, def | low,
			 "Minimum S/N selected", 0., 0.);
    parameters.addMember("selectSnMax",&selectSnMax, def | low,
			 "Maximum S/N selected", 1000., 0.);
    parameters.addMember("seed", &seed, def | low,
			 "Random number seed (0 uses time of day)", 0L, 0L);

    parameters.addMemberNoValue("LENSING:",0,
				"Applied lensing");
    parameters.addMember("g1",&g1, def,
			 "Applied g1", 0.01);
    parameters.addMember("g2",&g2, def,
			 "Applied g2", 0.);
#ifdef MAGNIFY
    parameters.addMember("mu",&mu, def,
			 "Applied magnification", 0.);
#endif

    parameters.addMemberNoValue("PRIOR:",0,
				"Characteristics of the sampled prior");
    parameters.addMember("priorSigmaCutoff",&priorSigmaCutoff, def | low,
			 "Maximum sigma range when sampling for prior", 6., 3.);
    parameters.addMember("priorSigmaStep",&priorSigmaStep, def | lowopen,
			 "Step size when sampling for prior", 1., 0.);
#ifdef USE_TREE
    parameters.addMember("priorSigmaBuffer",&priorSigmaBuffer, def | low,
			 "Buffer width of KdTreePrior (in sigma)", 1., 0.);
    parameters.addMember("nSample",&nSample, def | low,
			 "Number of templates sampled per target (0=all)", 30000L, 0L);
#endif

    parameters.addMemberNoValue("STATISTICS:",0,
				"Number of galaxies to use");
    parameters.addMember("nTarget",&nTarget, def | low,
			 "Number of target galaxies", 100000L, 100L);
    parameters.addMember("nTemplate",&nTemplate, def | low,
			 "Number of template galaxies", 10000L, 100L);
    parameters.addMember("minUniqueTemplates",&minUniqueTemplates, def | low,
			 "Minimum number of unique templates for target inclusion", 1, 1);

    parameters.addMemberNoValue("COMPUTING:",0,
				"Configure the computation, usually not needed");
    parameters.addMember("nThreads", &nThreads, def,
			 "Number of threads to use (-1=all)", -1);
    parameters.addMember("chunk", &chunk, def | low,
			 "Batch size dispatched to each thread", 100L, 1L);
    parameters.addMember("maxLeaf", &maxLeaf, def | low,
			 "Maximum number of templates in leaf nodes (0 to default)", 
			 0, 0);
    parameters.addMember("sampleWeights", &sampleWeights, def,
			 "Sample templates by weight (T) or number (F)?", true);
  }

  parameters.setDefault();
  
  try {

    int positionalArguments;
    try {
      // First read the command-line arguments so we know how many positional
      // arguments precede them.
      positionalArguments = parameters.setFromArguments(argc, argv);
    } catch (std::runtime_error &m) {
      // An error here might indicate someone entered "-help" or something
      cerr << usage << endl;
      cerr << "#---- Parameter defaults: ----" << endl;
      parameters.dump(cerr);
      quit(m,1);
    }
    for (int i=1; i<positionalArguments; i++) {
      // Open & read all specified input files
      ifstream ifs(argv[i]);
      if (!ifs) {
	cerr << "Can't open parameter file " << argv[i] << endl;
	cerr << usage << endl;
	exit(1);
      }
      try {
	parameters.setStream(ifs);
      } catch (std::runtime_error &m) {
	cerr << "In file " << argv[i] << ":" << endl;
	quit(m,1);
      }
    }
    // And now re-read the command-line arguments so they take precedence
    parameters.setFromArguments(argc, argv);

    cout << "#" << stringstuff::taggedCommandLine(argc,argv) << endl;
    parameters.dump(cout);

    const double noise=1.;	// White noise level
    const double sigmaW = 1.; // Weight function size
    Shear g;
    g.setG1G2(g1, g2);
#ifdef SHIFT
    double dxy=0.5*sigmaGMin;  // Displace initial galaxy guess by +-0.5 of sigma
#else
    double dxy=0.;  // No center variation
#endif

#ifdef _OPENMP
    if(nThreads>0) omp_set_num_threads(nThreads);
#endif

    ran::UniformDeviate<double> ud;
    if (seed > 0) ud.seed(seed);

    ran::GaussianDeviate<double> gd(ud); //e.g. float f = ud() to get a single number

    // Make galaxy distribution: same distribution for template & target
    GalaxyDistribution templateDistribution( wtSigma, sigmaE,
					     snMin, snMax, snGamma,
					     noise,
					     sigmaGMin, sigmaGMax,
					     ud, dxy);

    // Get the moment covariance matrix
    auto mcov = templateDistribution.getGal().getTarget(true).cov;
    // Make a noise generator for this covariance
    BC::Matrix allcov(BC::MXYSIZE, BC::MXYSIZE, FP(0));
    allcov.subMatrix(0,BC::MSIZE,0,BC::MSIZE) = mcov.m;
    if (!BC::FixCenter)
      allcov.subMatrix(BC::MSIZE,BC::MXYSIZE,BC::MSIZE,BC::MXYSIZE) = mcov.xy;
    MultiGauss<FP> momentNoise(allcov);

    double fluxMin = selectSnMin * sqrt(mcov.m(BC::MF,BC::MF));
    double fluxMax = selectSnMax * sqrt(mcov.m(BC::MF,BC::MF));

#ifdef SHIFT
    // Some quantities we'll need for building template arrays:
    FP fluxSigma = sqrt( mcov.m(BC::MF,BC::MF));
    FP xySigma = sqrt( 0.5*(mcov.xy(BC::MX,BC::MX)+mcov.xy(BC::MY,BC::MY)));
    FP xyMax = 2.*sigmaGMax; // Maximum range to allow centroid to vary for templates
#endif
    
#ifdef USE_TREE
    if(maxLeaf>0) Node<TemplateInfo<BC> >::setMaxElements(maxLeaf);

    // Create KdTree prior, use all samples if nSample=0
    // 3rd arg indicates target centers are assumed known.
    KDTreePrior<BC> prior(fluxMin, fluxMax,
			  mcov,
			  ud,
			  nSample,
			  false, // Makes this a probability prior, not selection
			  noiseFactor,
			  priorSigmaStep, priorSigmaCutoff, priorSigmaBuffer,
			  true, // Invariant covariance
			  true); // Noise is fixed on translation
    if(!sampleWeights) prior.setSampleWeights(false);
#else
    Prior<BC> prior(fluxMin, fluxMax,
		    mcov,
		    ud,
		    false, // Makes this a probability prior, not selection
		    noiseFactor,
		    priorSigmaStep, priorSigmaCutoff,
		    true, // Invariant covariance
		    true); // Noise is fixed on translation
#endif

    // Create another prior to calculate the selection probability
    auto selectionPriorPtr = new Prior<BC>(fluxMin, fluxMax,
					   mcov,
					   ud,
					   true, // Makes this calculate selection
					   noiseFactor,
					   priorSigmaStep, priorSigmaCutoff,
					   true, // Invariant covariance
					   true); // Noise is fixed on translation

    // Stop counting unique id's above this limit:
    prior.setUniqueCeiling(200);


    // Draw template galaxies - set timer
    Stopwatch timer;
    timer.start();

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
      for (long i = 0; i<nTemplate; i++) {
	auto gg = templateDistribution.getGal();
#ifdef SHIFT
	auto vtmpl = gg.getTemplateGrid(ud,
					xySigma,
					fluxSigma,
					fluxMin,
					priorSigmaStep,
					priorSigmaCutoff,
					xyMax);
	if (vtmpl.empty()) {
	  cerr << "WARNING: failed template #" << i << endl;
	  continue;
	}
	for (auto& tmpl : vtmpl) {
	  tmpl.nda /= nTemplate;
	  tmpl.id = i;
	  prior.addTemplate(tmpl, true);  // Include yflip copy;
	  selectionPriorPtr->addTemplate(tmpl, true);  // Include yflip copy;
	}
#else
	auto tmpl = gg.getTemplate();
	// In postage-stamp mode, the "density" of each template is 1./nTemplate
	tmpl.nda /= nTemplate;
	tmpl.id = i;
	prior.addTemplate(tmpl, true);  // Include yflip copy;
	selectionPriorPtr->addTemplate(tmpl, true);  // Include yflip copy;
#endif
      }
    } // end omp parallel section
    
    timer.stop();
    cout << "# Created " << prior.getNTemplates() << " templates in "
	 << timer << endl;
    timer.reset();

    timer.start();
    // Prepare the prior (builds tree, etc.)
    prior.prepare();
    selectionPriorPtr->prepare();
    timer.stop();
    cout << "# prepared in " << timer << endl;

    // Get the selection density
    Pqr<BCD> selectionPqr;
    {
      int junk;
      auto pqr = selectionPriorPtr->getPqr(TargetGalaxy<BC>(), junk, junk);
      selectionPqr += pqr;
    }
    // No longer need the selectionPrior, return memory.
    delete selectionPriorPtr;
    
    /////////////////////////////////////////////////////////////////
    // Now begin measuring targets.
    /////////////////////////////////////////////////////////////////

    Pqr<BCD> accumulator;  // Accumulate in double precision
    long nUsed = 0;  // Number of targets used
    long nPairs = 0;
    long nDeselect = 0;	// Number missing flux cuts

    chunk = MIN(chunk, nTarget);	//Number of galaxies to create per for loop
    long nLoops = nTarget / chunk;

    timer.reset();
    timer.start();

    int numThreads = 1;
#ifdef _OPENMP
#pragma omp parallel
#endif

    {

#ifdef _OPENMP
      numThreads = omp_get_num_threads();
#pragma omp for schedule(dynamic)
#endif
      for (long i=0; i<nLoops; i++) {
	Pqr<BCD> subAccum;
	long subNUsed=0;
	long subNPairs=0;
	long subNDeselect=0;
	for (long j=0; j<chunk; j++) {

	  // Create a sheared galaxy with noise on it
          auto gg = templateDistribution.getGal(g,mu);
	  GalaxyPlusNoise<BC> noisy(&gg, gd, &momentNoise);

#ifdef SHIFT
	  // Recenter the target
	  auto g2 = noisy.getNullXY(xyMax);
	  if (!g2) {
	    cerr << "NullXY failure" << endl;
	    ++subNDeselect;
	    continue;
	  }
	  auto targ = g2->getTarget(true);
	  delete g2;
#else
	  auto targ = noisy.getTarget(true);
#endif
	  int nTemplates;
	  int nUnique;
	  // Calculate the Pqr for this galaxy and add it in:
	  auto linearPqr = prior.getPqr(targ, nTemplates, nUnique);

	  if (linearPqr[BC::P]<0.) {
	    // Did not make selection cut
	    subNDeselect++;
	  } else {
	    subNPairs += nTemplates;
	    /**
#ifdef _OPENMP
#pragma omp critical(output)
#endif
	  cerr << nUnique
	       << " " << nTemplates
	       << " " << targ.mom.m[BC::MF] / sqrt(targ.cov.m(BC::MF,BC::MF))
		   << " " << linearPqr
		   << endl;
	    **/
	  
	    if (nUnique >= minUniqueTemplates) {
	      subAccum += linearPqr.neglog();
	      subNUsed++;
	    }
	  }
	} // end target loop
#ifdef _OPENMP
#pragma omp critical(accumulate)
#endif
	{
	  accumulator += subAccum;
	  nUsed += subNUsed;
	  nPairs += subNPairs;
	  nDeselect += subNDeselect;
	}
      } // end chunk loop
    } // end threaded region

    timer.stop();
    cout << "# Measured in " << timer 
	 << " " << nPairs / (numThreads * timer * 1e6)
	 << " Mpairs/core/sec" << endl;

    // Outputs:
    cout << "# Used " << nUsed << " deselected " << nDeselect 
	 << " rejected " << nTarget-nUsed-nDeselect << endl;
    cout << "# Pairs: " << nPairs / 1e9 << " billion" << endl;

    cerr << "# WITHOUT SELECTION: " << endl;
    cerr << "PQR: " << accumulator << endl;
    BCD::QVector gMean;
    BCD::RMatrix gCov;
    accumulator.getG(gMean, gCov);
    cerr << "g1: " << gMean[BC::G1] << " +- " << sqrt(gCov(BC::G1,BC::G1)) << endl;
    cerr << "g2: " << gMean[BC::G2] << " +- " << sqrt(gCov(BC::G2,BC::G2)) << endl;
#ifdef MAGNIFY
    cerr << "mu: " << gMean[BC::MU] << " +- " << sqrt(gCov(BC::MU,BC::MU)) << endl;
#endif

    // Include PQR for unselected galaxies
    cout << "# Selection PQR: " << selectionPqr << endl;
    double nTot = nUsed + nDeselect;
    linalg::DVector lens(BC::ND,0.);
    lens[BC::G1] = g1;
    lens[BC::G2] = g2;
#ifdef MAGNIFY
    lens[BC::MU] = mu;
#endif
    cout << "# Predict selection: " << selectionPqr(lens)
	 << " obtained " << (double) nUsed / (nUsed + nDeselect)
	 << " +- " << sqrt(nUsed * nDeselect / nTot ) / nTot
	 << endl;

    // Turn into de-selection 
    if (nDeselect > 0) {
      selectionPqr *= -1.;  
      selectionPqr[BC::P] += 1;
      accumulator += selectionPqr.neglog() * (double) nDeselect;
    }
    cout << "# WITH SELECTION: " << endl;
    cout << "PQR: " << accumulator << endl;
    accumulator.getG(gMean, gCov);
    cout << "g1: " << gMean[BC::G1] << " +- " << sqrt(gCov(BC::G1,BC::G1)) << endl;
    cout << "g2: " << gMean[BC::G2] << " +- " << sqrt(gCov(BC::G2,BC::G2)) << endl;
#ifdef MAGNIFY
    cout << "mu: " << gMean[BC::MU] << " +- " << sqrt(gCov(BC::MU,BC::MU)) << endl;
#endif
    
  } catch (std::runtime_error& m) {
    quit(m,1);
  }

  exit(0);
}
