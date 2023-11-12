// Integrate a set of target galaxies against a set of template galaxies.
// Each comes from objects listed in FITS binary table(s)

// #define MAGNIFY   // Define this macro to measure magnification
// #define CONC   // Define this macro to use the concentration (4th) moment

#include <ctime>

#include "BfdConfig.h"
#include "Pqr.h"
#include "Distributions.h"
#include "Shear.h"
#include "Moment.h"
#include "Galaxy.h"
#include "PqrAccumulator.h"
#include "Random.h"
#include "StringStuff.h"
#include "Prior.h"
#include "Pset.h"
#include "KdPrior.h"
#include "Stopwatch.h"
#include "MomentTable.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace bfd;

// Configure our classes
#ifdef CONC
const bool USE_CONC=true;
#else
const bool USE_CONC=false;
#endif

#ifdef MAGNIFY
const bool USE_MAG = true;
#else
const bool USE_MAG = false;
#endif

const bool FIX_CENTER = false;
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
  for (int i=0; i<BC::DSIZE; i++) lhs[i]+=rhs[i];
  return lhs;
}

typedef BC::FP FP;

const string usage = 
  "tableIntegrate: Integrate targets with moments given in FITS table against templates \n"
  "   given in a different FITS table using KdTree integration.\n"
  "   If no covariance column is in targetfile, all targets are assumed to have the \n"
  "   moment covariance matrix of their noise tier.  Targetfile must include at least\n"
  "   one noise tier."
  "   The *selectSN* parameter is a comma-separated list of values defining a series of\n"
  "     intervals in flux S/N that into which targets will be divided.  Zero means no bound,\n"
  "     and the default is that no selection cut is made. A single number will be taken\n"
  "	as a lower bound with no upper.\n"
  "   The *selectMF* parameter alternatively defines these intervals as absolute flux.\n"
  "   The *noiseFactor* parameter is a comma-separated list of noise inflation factors\n"
  "     that are applied to each interval, so there should be 1 fewer of these than\n"
  "     entries in the selectSN list.  Default is 1, no added noise.\n"
  "   The *useTiers* parameter is number of the noise tier of targets to process\n"
  "     in this run.  By default (-1), the first tier that matches the template file will be done.\n"
  "   *priorMatchTolerance* gives the maximum fractional deviation that flux or centroid sigma\n"
  "     used in building the templates can have from the nominal target values.\n"
  "   PQR values calculated for each target overwrite any PQR's already in the targetfile.\n"
  "   The *priorSigmaStep, priorSigmaMax* values are taken from templatefile unless overridden by\n"
  "     specification of positive values in the parameters.\n"
  "Usage: tableIntegrate [parameter file] [-parameter [=] value...]\n"
  "       Parameter files read in order of command line, then command-line parameters,\n"
  "       later values override earlier.\n"
  "stdin: none\n"
  "stdout: status info.\n"
  "Run with -help to get parameter list\n";

int main(int argc,
	 char *argv[])
{
  double priorSigmaMax;
  double priorSigmaStep;
  double priorSigmaBuffer;
  double priorMatchTolerance;

  long nSample;

  int nThreads;
  long chunk;
  int maxLeaf;
  long seed;
  bool sampleWeights;

  string targetFile;
  string templateFile;
  string noisetierFile;
  
  string noiseFactor;
  string addnoiseSN;
  int useTier;

  Pset parameters;
  {
    const int def=PsetMember::hasDefault;
    const int low=PsetMember::hasLowerBound;
    const int up=PsetMember::hasUpperBound;
    const int lowopen = low | PsetMember::openLowerBound;
    const int upopen = up | PsetMember::openUpperBound;

    parameters.addMember("targetFile",&targetFile, def,
			 "Input target moment table filename (will be updated)", "");
    parameters.addMember("noisetierFile",&noisetierFile, def,
			 "Input noise tier file", "");
    parameters.addMember("templateFile",&templateFile, def,
			 "Input template moment file", "");
    parameters.addMember("noiseFactor",&noiseFactor, def,
			 "Noise boost factor(s) for kernel smoothing", "1.");
    parameters.addMember("addnoiseSN",&addnoiseSN, def,
			 "S/N dividing noiseFactor ranges", "");

    parameters.addMemberNoValue("PRIOR:",0,
				"Characteristics of the sampled prior");
    parameters.addMember("priorSigmaBuffer",&priorSigmaBuffer, def | low,
			 "Buffer width of KdTreePrior (in sigma)", 1., 0.);
    parameters.addMember("priorMatchTolerance",&priorMatchTolerance, def | lowopen,
			 "Max allowed deviation of template from target sigmas", 0.05, 0.);
      
    parameters.addMember("nSample",&nSample, def | low,
			 "Number of templates sampled per target (0=all)", 30000L, 0L);
    parameters.addMember("maxLeaf", &maxLeaf, def | low,
			 "Maximum number of templates in leaf nodes (0 to default)", 
			 0, 0);
    parameters.addMember("sampleWeights", &sampleWeights, def,
			 "Sample templates by weight (T) or number (F)?", true);

    parameters.addMemberNoValue("COMPUTING:",0,
				"Configure the computation, usually not needed");
    parameters.addMember("nThreads", &nThreads, def,
			 "Number of threads to use (-1=all)", -1);
    parameters.addMember("chunk", &chunk, def | low,
			 "Batch size dispatched to each thread", 100L, 1L);
    parameters.addMember("seed", &seed, def | low,
			 "Random number seed (0 uses time of day)", 0L, 0L);
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

    /***************************
    Do some more checking of the parameters
    ***************************/
    
    // Do we have files?
    if (targetFile.empty() || templateFile.empty() || noisetierFile.empty()) {
      cerr << "ERROR: Need to specify targetFile, noisetierFile, and templateFile parameters" << endl;
      exit(1);
    }

    Stopwatch timer;

    // Open the target file
    timer.reset();
    timer.start();
    TargetTable<BC> targets(targetFile,true);  // Update this file
    timer.stop();
    cerr << "# Read targets in " << timer << endl;
    
    // Add a PQR column to it
    targets.checkPqrColumn();

    // Add nUnique column to it
    targets.checkNUniqueColumn();
    
    // Open the noisetier file
    NoiseTierCollection<BC> noisetiers(noisetierFile);
    cerr << "# Read noise tiers" << endl;

    // Open the template file
    timer.reset();
    timer.start();
    TemplateTable<BC> templates(templateFile);
    timer.stop();
    cerr << "# Read templates in " << timer << endl;

    // Get our noise tier.
    int useTier = templates.getTier();
    cerr << "# Using noise tier " << useTier << endl;
    // Get pointer to our chosen NoiseTier
    auto ntptr = noisetiers[useTier];
    // Get the specs for template spacing etc, 
    auto tsTier = ntptr->needsTemplateSpecs();

    // Are templates good enough?
    auto tsTemplate = templates.getTemplateSpecs();
    if (!tsTemplate.goodEnoughFor(tsTier,
				  priorMatchTolerance)) {
      cerr << "ERROR: Template sampling not a sufficient match to tier specs" << endl;
      exit(1);
    }

    priorSigmaStep = tsTemplate.sigmaStep;
    priorSigmaMax = tsTemplate.sigmaMax;

    if (targets.weightN() != tsTemplate.weightN ||
	abs(targets.weightSigma()/tsTemplate.weightSigma-1) > 0.0001) {
      cerr << "ERROR: Mismatch between target and template weight functions" << endl;
      cerr << "N: " << targets.weightN() << " vs " << tsTemplate.weightN << endl;
      cerr << "Sigma: " << targets.weightSigma() << " vs " << tsTemplate.weightSigma << endl;
      exit(1);
    }

    // Acquire a nominal covariance matrix from the noise tier
    MomentCovariance<BC> mcov = ntptr->nominalCov();
    double sigmaF = sqrt(mcov.m(BC::MF,BC::MF));

    // Next parse the add-noise ranges

    // Pull apart the selection ranges and check them
    vector<double> vFluxIn;
    for (auto sn : stringstuff::split(addnoiseSN, ',')) {
      stringstuff::stripWhite(sn);
      if (sn.empty()) continue;  // Skip empty regions
      vFluxIn.push_back(atof(sn.c_str()) * sigmaF);
    }
    vector<double> vNoiseIn;
    for (auto nf : stringstuff::split(noiseFactor, ',')) {
      vNoiseIn.push_back(atof(nf.c_str()));
    }
    if (vFluxIn.size() != vNoiseIn.size()-1) {
      cerr << "ERROR: noiseFactor and addnoiseSN lists have unmatched lengths " << endl;
      exit(1);
    }

    // Build vector of flux range bounds and noise factors
    vector<double> vFlux;
    vector<double> vNoise;
    int iMin = 0;
    // Get min flux of first range at fluxMin
    if (tsTier.fluxMin <=0.) {
	// No minimum flux
	vFlux.push_back(0.);
      } else {
        // Determine which entry of vFluxIn table is just *above* fluxMin
        for (iMin=0; iMin<vFluxIn.size() && vFluxIn[iMin]<tsTier.fluxMin; iMin++) ;
	vFlux.push_back(tsTier.fluxMin);
    }
    // Get noise factors and upper bounds for each range
    for ( ; iMin <= vFluxIn.size(); iMin++) {
      vNoise.push_back(vNoiseIn[iMin]);
      if (iMin==vFluxIn.size() || (tsTier.fluxMax>0. &&
				   tsTier.fluxMax <= vFluxIn[iMin])) {
	// This is last range, upper lim is fluxMax
	vFlux.push_back( max(tsTier.fluxMax, 0.));
      } else {
	// Upper lim is the next S/N split value
	vFlux.push_back(vFluxIn[iMin]);
      }
    }
      
    // Check for increasing ranges; except that the last value could
    // be zero to indicate unbounded upper range
    int nRanges = vFlux.size()-1;
    if (vFlux[0] <0) vFlux[0] = 0.;
    for (int i=0; i<nRanges-1; i++) {
      if (i==nRanges && vFlux[i]<=0.) {
	// Unbounded upper bin is ok.
	continue;
      } else if (vFlux[i+1] <= vFlux[i]) {
	cerr << "ERROR: range bounding flux values are not increasing" << endl;
	exit(1);
      }
    }

    {
      // Record this operation in header of target catalog
      std::time_t rawtime;
      struct std::tm* timeinfo;
      char buffer[40];
      std::time (&rawtime);
      timeinfo = std::gmtime(&rawtime);
      strftime(buffer, 40 , "%F %T", timeinfo);
      string t = buffer;
      ostringstream oss;
      oss << t
	  << ": Calculating PQR for tier " <<useTier
	  << " from templates " << templateFile;
      targets.addHistory(oss.str());
    }
    
    //****************
    // Configure the calculations
    
#ifdef _OPENMP
    if(nThreads>0) omp_set_num_threads(nThreads);
#endif

    if(maxLeaf>0) Node<TemplateInfo<BC> >::setMaxElements(maxLeaf);

    ran::UniformDeviate<double> ud;
    if (seed > 0) ud.seed(seed);

    // Loop through each selection range:
    for (int iSN=0; iSN<vFlux.size()-1; iSN++) {
      cout << "# ****S/N range " << vFlux[iSN]/sigmaF << " - " << vFlux[iSN+1]/sigmaF << endl;
      double fluxMin = vFlux[iSN];
      double fluxMax = vFlux[iSN+1];

      // Find all the targets that are selectable here
      // Skip all targets with non-positive jacobians!
      vector<long> doThese;
      XYJacobian<BC> jacobianFunction;
      for (int i=0; i<targets.size(); i++) {
	const auto& moments = targets.getTarget(i).mom.m;
	double flux = moments[BC::MF];
	if (targets.getTier(i)==useTier &&
	    (fluxMin==0. || flux >=fluxMin) &&
	    (fluxMax==0. || flux <fluxMax) &&
	    jacobianFunction(moments)>0) {
	  doThese.push_back(i);
	}
      }

      cerr << "# Galaxies in this S/N range: " << doThese.size() << endl;
    
      // Do not need to do anything else for the selection range if no galaxies to measure.
      if (doThese.empty())
	continue;

      // Create KdTree prior, use all samples if nSample=0
      KDTreePrior<BC> prior(fluxMin, fluxMax,
			    mcov,
			    ud,
			    nSample,
			    false, // Makes this a probability prior, not selection
			    vNoise[iSN],
			    priorSigmaStep, priorSigmaMax, priorSigmaBuffer,
			    targets.hasSharedCov()); 

      if(!sampleWeights) prior.setSampleWeights(false);

      // Stop counting unique id's above this limit:
      prior.setUniqueCeiling(200);

      // Read the template galaxies
      timer.start();

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (long i = 0; i<templates.size(); i++) {
	  auto tmpl = templates.getTemplate(i);
	  prior.addTemplate(tmpl, true);  // Include yflip copy;
	}
      } // end omp parallel section
      timer.stop();
      cout << "# Ingested " << prior.getNTemplates() << " templates in "
	   << timer << endl;

      timer.reset();
      timer.start();
      // Prepare the prior (builds tree, etc.)
      prior.prepare();
      timer.stop();
      cerr << "# prepared in " << timer << endl;

      /////////////////////////////////////////////////////////////////
      // Now begin measuring targets.
      /////////////////////////////////////////////////////////////////

      long nPairs = 0;

      long nTarget = doThese.size();
      chunk = MIN(chunk, nTarget);	//Number of galaxies to create per for loop
      long nLoops = (nTarget-1) / chunk + 1;

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
	  long first = i*chunk;
	  long last = MIN((i+1)*chunk, nTarget);

#ifdef _OPENMP
#pragma omp critical(io)
	  cerr << "#--Thread " << omp_get_thread_num() << " starting " << first << "->" << last << endl;
#endif
	  // Hold the results internal to the thread
	  vector<Pqr<BC>> subPqr(last-first);  // ?? Use special Eigen vector class?
	  vector<int> subNUnique(last-first); 
	  long subNPairs=0;
	  
	  for (long j=first; j<last; j++) {
	    auto targ = targets.getTarget(doThese[j]);
	    int nTemplates;
	    int nUnique;
	    // Calculate the Pqr for this galaxy and add it in:
	    auto linearPqr = prior.getPqr(targ, nTemplates, nUnique);

	    if (linearPqr[BC::P]>0.) {
	      // A selected target, record results:
	      subNPairs += nTemplates;
	      subPqr[j-first] = linearPqr;  
	      subNUnique[j-first] = nUnique; 
	    }
	  } // end target loop
#ifdef _OPENMP
#pragma omp critical(accumulate)
#endif
	  {
	    for (long i=0; i<subPqr.size(); i++) {
	      if (subPqr[i][BC::P] > 0.) {
		// Add Pqr to existing values
		targets.addPqr(subPqr[i], doThese[i+first]);
		// And nUnique
		targets.setNUnique(subNUnique[i], doThese[i+first]);		
	      }
	    }
	    nPairs += subNPairs;
	  } // End accumulation critical region

	} // end chunk loop
      } // end threaded region

      timer.stop();
      cout << "# Measured in " << timer 
	   << " sec, " << nPairs / (numThreads * timer * 1e6)
	   << " Mpairs/core/sec" << endl;
    } // end selection range loop
    
  } catch (std::runtime_error& m) {
    quit(m,1);
  }

  exit(0);
}
