// Test of BFD using direct-to-k-space drawing of galaxies.
// Version with no recentering

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
#include "SbPsf.h"

// Undefine this to use brute-force integration.
// Which would be pretty foolish.
#define USE_TREE

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace bfd;
const string usage = 
  "sbTest4: Estimate shear with BFD method using galaxies drawn in k space, no recentering.\n"
#ifdef USE_TREE
  "            KdTree integration.\n"
#else
  "            Brute-force integration\n"
#endif
  "            Rotation to align templates with targets to within nSigma.\n"
  "            Weight function sigma defined as 1.\n"
  "Usage: sbTest4 [parameter file] [-parameter [=] value...]\n"
  "       Parameter files read in order of command line, then command-line parameters,\n"
  "       later values override earlier.\n"
  "stdin: none\n"
  "     Run with -help to get parameter list\n"
  "stdout: The summed P,Q,R of the targets and the estimated shear & uncertainty";

//**const int UseMoments = USE_E_MOMENT+USE_FLUX_MOMENT+USE_SIZE_MOMENT;
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

// Class that produces Galaxies from specified distribution
class galaxyDistribution{
public:
  galaxyDistribution(const KWeight& kw_, const Psf& psf_,
		     double sigmaE, 
		     double snMin, double snMax, double snGamma,
		     double noise_, 
		     double sigmaGMin_, const double sigmaGMax_,
		     const double dxy_, ran::UniformDeviate& ud_): kw(kw_), psf(psf_),
								   ud(ud_),
								   eDist(sigmaE, ud_),
								   snDist(snMin,snMax,snGamma,ud),
								   noise(noise_),
								   sigmaGMin(sigmaGMin_),
								   sigmaGMax(sigmaGMax_),
								   dxy(dxy_) {
    // Determine flux scaling to S/N for average size, circular galaxy
    Ellipse ell(Shear(0.,0.), log(0.5*(sigmaGMin+sigmaGMax)), Position<double>(0.,0.));
    double flux = 1.;
    TemplateGalaxy<UseMoments>* gg = ellipse2galaxy(kw, psf, flux, ell, noise, ud, false);
    snScale = sqrt(gg->getCov()(MI::FLUX,MI::FLUX)) / gg->getMoments()[MI::FLUX];
    /**/cerr << "Setting snScale to " << snScale << endl;
    delete gg;
}

  // Galaxy sampling method
  // Default is no shear.  For shear, put shear as arg)
  // Set fixCenter = false to have galaxy centered at origin.
  TemplateGalaxy<UseMoments>* getGal(Shear g=Shear(0.,0.), bool isTemplate=false, bool fixCenter=false) {
    double e1, e2;
    double sn;
    double flux;
    double sigmaG;
    Position<double> ctr(0.,0.);
    TemplateGalaxy<UseMoments>* gg = 0;
#ifdef _OPENMP
#pragma omp critical(random)
#endif
    {
      eDist.sample(e1, e2);
      sn = snDist.sample();
      sigmaG = sigmaGMin + ud() * (sigmaGMax - sigmaGMin);
      if (!(fixCenter || isTemplate)) {
	ctr.x = (2*ud()-1.)*dxy;
	ctr.y = (2*ud()-1.)*dxy;
      }

      // The distortion applied to each target:
      Ellipse distort(g, 0., Position<double>(0.,0.)); 
      Ellipse ell(Shear(e1,e2), log(sigmaG), ctr);
      flux = sn * snScale;
      gg = ellipse2galaxy(kw, psf, flux, distort + ell, noise, ud, !isTemplate);
    }

    return  gg;
  } 

private:
  const KWeight& kw;
  const Psf& psf;
  ran::UniformDeviate& ud;
  BobsEDist eDist;	// The parent ellipticity distribution
  PowerLawDistribution snDist;
  double noise;
  double sigmaGMin;
  double sigmaGMax;
  double dxy;
  double snScale; // Conversion factor from S/N to flux.
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
  double sigmaW;
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
    parameters.addMember("sigmaW",&sigmaW, def | lowopen,
			 "Weight function size", 2., 0.);

    parameters.addMember("noiseFactor",&noiseFactor, def | low,
			 "Noise boost factor for kernel smoothing", 0., 0.);
    parameters.addMember("selectSnMin",&selectSnMin, def | low,
			 "Minimum S/N selected", 0., 0.);
    parameters.addMember("selectSnMax",&selectSnMax, def | low,
			 "Maximum S/N selected", 1000., 0.);
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
    const double sigmaPSF = 1.; // PSF size
    Shear g;
    g.setG1G2(g1, g2);
    double dxy=0.;

#ifdef _OPENMP
    if(nThreads>0) omp_set_num_threads(nThreads);
#endif

    ran::UniformDeviate ud;
    if (seed > 0) ud.Seed(seed);
    ran::GaussianDeviate gd(ud); //e.g. float f = ud() to get a single number

    const GaussianWeight kw(sigmaW);
    sbp::SBGaussian psf0(sigmaPSF);
    SbPsf psf(psf0);
    cerr << "psf kmax: " << psf.kMax() << endl;


    // Make galaxy distribution: same distribution for template & target
    galaxyDistribution templateDistribution( kw, psf, sigmaE,
					     snMin, snMax, snGamma,
					     noise,
					     sigmaGMin, sigmaGMax,
					     dxy, ud);

    // Get the galaxy covariance matrix and build a noise generator
    MomentCovariance<UseMoments> mcov;
    {
      TemplateGalaxy<UseMoments>* gg = templateDistribution.getGal();
      mcov = gg->getCov();
      /**/cerr << "Covariance:\n" << mcov << endl;
      delete gg;
    }
        
    double fluxMin;
    double fluxMax;
    if (MI::UseFlux) {
      fluxMin = selectSnMin * sqrt(mcov(MI::FLUX,MI::FLUX));
      fluxMax = selectSnMax * sqrt(mcov(MI::FLUX,MI::FLUX));
    }

    /**/cerr << "Fluxmin,max: " << fluxMin << " " << fluxMax << endl;
    
#ifdef USE_TREE
    if(maxLeaf>0) Node<MomentInfo<UseMoments> >::setMaxElements(maxLeaf);

    // Create KdTree prior, use all samples if nSample=0
    KDTreePrior<UseMoments> prior(fluxMin, fluxMax, true,
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

    // Set sampling parameters
    prior.setSamplingRange(priorSigmaCutoff, priorSigmaStep);

    // Add noise if desired
    if (MI::UseFlux && noiseFactor > 1.)
      prior.addNoiseFactor(noiseFactor, ud);

    // Stop counting unique id's above this limit:
    prior.setUniqueCeiling(200);

    // Save moments of the template
    MI::MVector mtempl;

    Stopwatch timer;
    timer.start();
    // Draw template galaxies (no noise, fixed centers)
    for (long i = 0; i<nTemplate; i++) {
      TemplateGalaxy<UseMoments>* gg = templateDistribution.getGal(Shear(0.,0.), true);
      // Add galaxy to template, do not put in parity flip
      prior.addTemplate( *gg, ud, dxy, 1., false, i);
      mtempl = gg->getMoments();
      delete gg;
    }
    /**/cerr << "template moments: " << mtempl << endl;
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

    // Will also get an empirical covariance matrix for when we have identical sources
    DVector msum(MI::N,0.);
    DMatrix covsum(MI::N,MI::N,0.);
    long nsum=0;
    
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
	  // Create the sheared, noisy galaxy
	  TemplateGalaxy<UseMoments>* gg = templateDistribution.getGal(g,false,true); // fix center

	  // Accumulate difference from template
	  MI::MVector m = gg->getMoments();
	  DVector dm(MI::N);
	  for (int i=0; i<MI::N; i++) dm[i] = m[i] - mtempl[i];
	  msum += dm;
	  covsum += dm^dm;
	  nsum++;
	  
	  Moments<UseMoments> mobs(gg->getMoments());
	  delete gg;

	  // Abort this galaxy if it does not meet the flux selection criterion
	  if (MI::UseFlux && (mobs[MI::FLUX] < fluxMin || mobs[MI::FLUX] > fluxMax)) {
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

    // Calculate covariance matrix
    msum /= (double) nsum;
    covsum /= (double) nsum;
    covsum -= msum ^ msum;
    cout << "Mean moment difference: " << msum << endl;
    cout << "Covariance matrix unshifted: \n" << covsum << endl;

    // Give std error a shear estimate without accounting for deselections:
    {
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
    }
    // Include PQR for unselected galaxies
    Pqr pSelect = prior.selectionProbability(mcov);
    cout << "# pSelect: " << pSelect << endl;
    cout << "# selection at g: " << pSelect[Pqr::P]
      + g1 * pSelect[Pqr::DG1]
      + g1 * pSelect[Pqr::DG2]
      + 0.5 * g1 * g1 * pSelect[Pqr::D2G1G1]
      + 0.5 * g1 * g2 * pSelect[Pqr::D2G1G2]
      + 0.5 * g2 * g2 * pSelect[Pqr::D2G2G2]
	 << endl;
    cout << "# Actual selection rate: " << (nTarget - nDeselect) / (double) nTarget
	 << endl;

    for (int i=0; i<Pqr::SIZE; i++) {
      if (i==Pqr::P)
	pSelect[i] = 1 - pSelect[i];
      else
	pSelect[i] *= -1.;
    }
    accumulator += pSelect.neglog() * (double) nDeselect;
    cout << "# *** with non-detections: ***" << endl;

    // Outputs:
    cout << "# Used " << nUsed << " deselected " << nDeselect 
	 << " rejected " << nTarget-nUsed-nDeselect << endl;
    cout << "# Pairs: " << nPairs / 1e9 << " billion" << endl;
    timer.stop();
    cout << "# Finished in " << timer << endl;

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
