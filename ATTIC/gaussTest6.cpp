// Gaussian test with centroid shifting

#include "Pqr.h"
#include "Distributions.h"
#include "Shear.h"
#include "Moments.h"
#include "Galaxy.h"
#include "Random.h"
#include "StringStuff.h"
#include "Prior.h"
#include "KdPrior.h"
#include "Pset.h"
#include "Stopwatch.h"

// Undefine this to use brute-force integration.
// Which would be pretty foolish.
#define USE_TREE

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace bfd;
const string usage = 
  "gaussTest6: Estimate shear with BFD method using Gaussian galaxies and weight function.\n"
  "            Moments are calculated analytically.\n"
#ifdef USE_TREE
  "            KdTree integration.\n"
#else
  "            Brute-force integration\n"
#endif
  "            Centroid shifting and rotation to align templates with targets to within nSigma.\n"
  "            Weight function sigma defined as 1.\n"
  "Usage: gaussTest6 [parameter file] [-parameter [=] value...]\n"
  "       Parameter files read in order of command line, then command-line parameters,\n"
  "       later values override earlier.\n"
  "stdin: none\n"
  "     Run with -help to get parameter list\n"
  "stdout: The summed P,Q,R of the targets and the estimated shear & uncertainty";

const int UseMoments = USE_ALL_MOMENTS;
typedef MomentIndices<UseMoments> MI;

Galaxy<UseMoments>*
newtonShift(const Galaxy<UseMoments>& gin, 
	    const MI::MVector& addnoise,
	    double& cx, double& cy,
	    const MomentCovariance<UseMoments>& cov,
	    int iterations=2) {
  // Do fixed number of Newton iteration to bring the xy moments near zero:
  cx = cy = 0.;
  const Galaxy<UseMoments>* gbase=&gin;
  Galaxy<UseMoments>* shifted=0;
  for (int iter=0; iter<iterations; iter++) {
    DVector2 dm;
    dm[0] = gbase->getMoments()[MI::CX] + addnoise[MI::CX];
    dm[1] = gbase->getMoments()[MI::CY] + addnoise[MI::CY];
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
  dm[0] = gbase->getMoments()[MI::CX] + addnoise[MI::CX];
  dm[1] = gbase->getMoments()[MI::CY] + addnoise[MI::CY];
  /** cerr << iter 
	   << " ctr " << cx << " " << cy
	   << " moments " << dm 
	    << " sigma " << dm / (double) sqrt(cov(MI::CX,MI::CX)) 
	   << endl; 
  /**/
  }
  return shifted;
}


GaussianGalaxy<UseMoments> 
ellipse2galaxy(const KWeight& kw, double flux, const Ellipse& el, double noise) {
  double e =  el.getS().getE();
  double beta = el.getS().getBeta();
  Position<double> c = el.getX0();
  double sigma = exp(el.getMu()) * pow(1-e*e, -0.25);
  return GaussianGalaxy<UseMoments>(kw, flux, sigma, e, beta, c.x, c.y, noise);
}

// Class that produces Galaxies from specified distribution
class galaxyDistribution{
public:
  galaxyDistribution(const KWeight& kw_, double sigmaE, 
		     double snMin, double snMax, double snGamma,
		     double noise_, 
		     double sigmaGMin_, const double sigmaGMax_, 
		     const double dxy_, ran::UniformDeviate& ud_): kw(kw_), 
								   ud(ud_),
								   eDist(sigmaE, ud_),
								   snDist(snMin,snMax,snGamma,ud),
								   noise(noise_),
								   sigmaGMin(sigmaGMin_),
								   sigmaGMax(sigmaGMax_),
								   dxy(dxy_) {}

  // Galaxy sampling method
  // Default is no shear.  For shear, put shear as arg)
  // Set fixCenter = false to have galaxy centered at origin.
  GaussianGalaxy<UseMoments> getGal(Shear g=Shear(0.,0.), bool fixCenter=false) {
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
      if (!fixCenter) {
	ctr.x = (2*ud()-1.)*dxy;
	ctr.y = (2*ud()-1.)*dxy;
      }
    }

    // The distortion applied to each target:
    Ellipse distort(g, 0., Position<double>(0.,0.)); 
    Ellipse ell(Shear(e1,e2), log(sigmaG), ctr);
    flux = sn * sqrt(4*PI*sigmaG*sigmaG*noise);
    GaussianGalaxy<UseMoments> gg = ellipse2galaxy(kw, flux, distort + ell, noise);

    return  gg;
  } 

private:
  const KWeight& kw;
  ran::UniformDeviate& ud;
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
			 "Maximum S/N selected", 0., 0.);
    parameters.addMember("seed", &seed, def | low,
			 "Random number seed (0 uses time of day)", 0L, 0L);

    parameters.addMemberNoValue("SHEAR:",0,
				"Applied shear");
    parameters.addMember("g1",&g1, def,
			 "Applied g1", 0.01);
    parameters.addMember("g2",&g2, def,
			 "Applied g2", 0.);

    parameters.addMemberNoValue("PRIOR CONFIGURATION:",0,
				"Characteristics of the sampled prior");
    parameters.addMember("priorSigmaCutoff",&priorSigmaCutoff, def | low,
			 "Maximum sigma range when sampling for prior", 5., 3.);
    parameters.addMember("priorSigmaStep",&priorSigmaStep, def | lowopen,
			 "Step size when sampling for prior", 0.5, 0.);
#ifdef USE_TREE
    parameters.addMember("priorSigmaBuffer",&priorSigmaBuffer, def | low,
			 "Buffer width of KdTreePrior (in sigma)", 1., 0.);
    parameters.addMember("nSample",&nSample, def | low,
			 "Number of templates sampled per target (0=all)", 50000L, 0L);
#endif

    parameters.addMemberNoValue("STATISTICS:",0,
				"Number of galaxies to use");
    parameters.addMember("nTarget",&nTarget, def | low,
			 "Number of target galaxies", 100000L, 100L);
    parameters.addMember("nTemplate",&nTemplate, def | low,
			 "Number of template galaxies", 10000L, 1L);
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
    double dxy=4.;

#ifdef _OPENMP
    if(nThreads>0) omp_set_num_threads(nThreads);
#endif

    ran::UniformDeviate ud;
    if (seed > 0) ud.Seed(seed);
    ran::GaussianDeviate gd(ud); //e.g. float f = ud() to get a single number

    const GaussianWeight kw(sigmaW);

    // Make galaxy distribution: same distribution for template & target
    galaxyDistribution templateDistribution( kw, sigmaE,
					     snMin, snMax, snGamma,
					     noise,
					     sigmaGMin, sigmaGMax,
					     dxy, ud);

    // Get the galaxy covariance matrix and build a noise generator
    MomentCovariance<UseMoments> mcov;
    mcov = templateDistribution.getGal().getCov();
    
    MI::MSymMatrix cov(MI::N);
    for (int i=0; i<cov.size(); i++)
      for (int j=0; j<=i; j++)
	cov(i,j) = mcov(i,j);

    MultiGauss<MI::Type> momentNoise(cov);

    double fluxMin;
    double fluxMax;
    if (MI::UseFlux) {
      fluxMin = selectSnMin * sqrt(mcov(MI::FLUX,MI::FLUX));
      fluxMax = selectSnMax * sqrt(mcov(MI::FLUX,MI::FLUX));
    }

#ifdef USE_TREE
    if(maxLeaf>0) Node<MomentInfo<UseMoments> >::setMaxElements(maxLeaf);

    // Create KdTree prior, use all samples if nSample=0
    KDTreePrior<UseMoments> prior(fluxMin, fluxMax, false,
				  nSample, ud, priorSigmaBuffer,MID,
				  nSample<=0 ? true : false);
    if(!sampleWeights) prior.setSampleWeights(false);
#else
    SampledPrior<UseMoments> prior(fluxMin, fluxMax, false);
#endif

    // Prepare the prior
    prior.setNominalCovariance(mcov);

    // Exploit all covariances identical:
    prior.setInvariantCovariance();

    // Note that noise does not vary as centroid moves
    prior.setShiftingNoise(false);
      
    // Set sampling parameters
    prior.setSamplingRange(priorSigmaCutoff, priorSigmaStep);

    // Add noise if desired
    if (MI::UseFlux && noiseFactor > 1.)
      prior.addNoiseFactor(noiseFactor, ud);

    // Stop counting unique id's above this limit:
    prior.setUniqueCeiling(200);

    Stopwatch timer;
    timer.start();
    // Draw template galaxies, holding centroid fixed
    for (long i = 0; i<nTemplate; i++) {
      GaussianGalaxy<UseMoments> gg = templateDistribution.getGal(Shear(0.,0.), true);
      // Add galaxy to template, do not put in parity flip
      prior.addTemplate( gg, ud, dxy, 1., false, i);
    }
    timer.stop();
    cout << "# Created " << prior.getNTemplates() << " templates in "
	 << timer << endl;
    timer.reset();

    timer.start();
    // Prepare the prior (builds tree, etc.)
    prior.prepare();
    timer.stop();
    cout << "# prepared in " << timer << endl;
    
    /////////////////////////////////////////////////////////////////
    // Now begin measuring targets.
    /////////////////////////////////////////////////////////////////

    Pqr accumulator;
    long nUsed = 0;
    long nPairs = 0;
    long nDeselect = 0;	// Number missing flux cuts

    chunk = MIN(chunk, nTarget);	//Number of galaxies to create per for loop
    long nLoops = nTarget / chunk;
    
    timer.reset();
    timer.start();
#ifdef _OPENMP
#pragma omp parallel
#endif

    {

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
      for (long i=0; i<nLoops; i++) {
	Pqr subAccum;
	long subNUsed=0;
	long subNPairs=0;
	long subNDeselect=0;
	for (long j=0; j<chunk; j++) {
	  double e1, e2;
	  MI::MVector addnoise;
	  MI::MVector m;

#ifdef _OPENMP
#pragma omp critical(random)
#endif
	  {
	    // Must block other threads in this block until we make parallel random-number generators
	    addnoise=momentNoise.sample(gd);
	  }

	  // Create the sheared, noiseless galaxy, also holding center fixed at origin
	  GaussianGalaxy<UseMoments> gg = templateDistribution.getGal(g, true);

	  // Recenter, a bit kludgy as the noise is not part of a Galaxy
	  double cx,cy;
	  Galaxy<UseMoments>* shifted = newtonShift(gg, addnoise, cx, cy, mcov, 2);

	  Moments<UseMoments> mobs(shifted->getMoments() + addnoise);
	  delete shifted;

	  // Abort this galaxy if it does not meet the flux selection criterion
	  if (MI::UseFlux &&
	      ( (fluxMin!=0. && mobs[MI::FLUX] < fluxMin) ||
		(fluxMax!=0. && mobs[MI::FLUX] > fluxMax))) {
	    subNDeselect++;
	    continue;
	  }

	  // Calculate the Pqr for this galaxy and add it in:
	  int nTemplates;
	  int nUnique;
	  Pqr linearPqr = prior.getPqr2(mobs, mcov, nTemplates, nUnique);
	  subNPairs += nTemplates;
	  if (nUnique >= minUniqueTemplates) {
	    subAccum += linearPqr.neglog();
	    subNUsed++;
	  }
	}
#ifdef _OPENMP
#pragma omp critical(accumulate)
#endif
	{
	  accumulator += subAccum;
	  nUsed += subNUsed;
	  nPairs += subNPairs;
	  nDeselect += subNDeselect;
	}
      }
    }

    timer.stop();
    cout << "# Finished in " << timer << endl;
    cout << "# Used " << nUsed << " deselected " << nDeselect 
	 << " rejected " << nTarget-nUsed-nDeselect << endl;
    cout << "# Pairs: " << nPairs / 1e9 << " billion" << endl;
    cerr << "# *** NO deselection correction:" << endl;
    cerr << "PQR: " << accumulator[Pqr::P] 
	 << " " << accumulator[Pqr::DG1]
	 << " " << accumulator[Pqr::DG2]
	 << " " << accumulator[Pqr::D2G1G1]
	 << " " << accumulator[Pqr::D2G2G2]
	 << " " << accumulator[Pqr::D2G1G2]
	 << endl;
    Pqr::GVector gMean;
    Pqr::GMatrix gCov;
    accumulator.getG(gMean, gCov);
    cerr << "g1: " << gMean[Pqr::G1] << " +- " << sqrt(gCov(Pqr::G1,Pqr::G1))
	 << " g2: " << gMean[Pqr::G2] << " +- " << sqrt(gCov(Pqr::G2,Pqr::G2))
	 << " r: " << gCov(Pqr::G1,Pqr::G2) / sqrt(gCov(Pqr::G2,Pqr::G2)*gCov(Pqr::G1,Pqr::G1))
	 << endl;

    // Include PQR for unselected galaxies
    Pqr pSelect = prior.selectionProbability(mcov);
    cout << "# pSelect: " << pSelect << endl;
    double nTot = nUsed + nDeselect;
    cout << "# Predict selection: " << pSelect[Pqr::P]
      + g1 * pSelect[Pqr::DG1]
      + g1 * pSelect[Pqr::DG2]
      + 0.5 * g1 * g1 * pSelect[Pqr::D2G1G1]
      + 1.0 * g1 * g2 * pSelect[Pqr::D2G1G2]
      + 0.5 * g2 * g2 * pSelect[Pqr::D2G2G2]
	 << " obtained " << (double) nUsed / (nUsed + nDeselect)
	 << " +- " << sqrt(nUsed * nDeselect / nTot ) / nTot
	 << endl;

    pSelect[Pqr::P] = 1 - pSelect[Pqr::P];
    for (int i=1; i<Pqr::SIZE; i++) 
      pSelect[i] *= -1.;
    accumulator += pSelect.neglog() * (double) nDeselect;
    cout << "# *** with non-detections: ***" << endl;
    cout << "PQR: " << accumulator[Pqr::P] 
	 << " " << accumulator[Pqr::DG1]
	 << " " << accumulator[Pqr::DG2]
	 << " " << accumulator[Pqr::D2G1G1]
	 << " " << accumulator[Pqr::D2G2G2]
	 << " " << accumulator[Pqr::D2G1G2]
	 << endl;
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
