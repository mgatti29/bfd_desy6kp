// Calculate the selection PQR for all the covariance nodes
// specified in a noise tier extension.
// Needs a template table, in standard FITS table formats.

// #define MAGNIFY   // Define this macro to measure magnification
// #define CONC   // Define this macro to use the concentration (4th) moment

#include <ctime>

#include "BfdConfig.h"
#include "Pqr.h"
#include "Distributions.h"
#include "Shear.h"
#include "Moment.h"
#include "Galaxy.h"
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
  "tierSelection:  Calculate the selection PQR for a given noise tier\n"
  "   using a template table built for that tier.\n"
  "   *priorMatchTolerance* gives the maximum fractional deviation that\n"
  "     flux or centroid sigma used in building the templates can have\n"
  "     from the nominal tier values.\n"
  "The calculated selection PQR's for the tier's covariance nodes are\n"
  "     written to the table.\n"
  "Usage: tierSelection <noisetier file> <template file> [-parameter [=] value...]\n"
  "stdin: none\n"
  "     Run with -help to get parameter list\n"
  "stdout: parameter summary \n";

int main(int argc,
	 char *argv[])
{
  double priorSigmaMax;
  double priorSigmaStep;
  double priorMatchTolerance;
  int nThreads;

  /** Nothing random or threaded here 
  long chunk;
  long seed;
  bool sampleWeights;
  **/

  
  Pset parameters;
  {
    const int def=PsetMember::hasDefault;
    const int low=PsetMember::hasLowerBound;
    const int up=PsetMember::hasUpperBound;
    const int lowopen = low | PsetMember::openLowerBound;
    const int upopen = up | PsetMember::openUpperBound;

    parameters.addMember("priorMatchTolerance",&priorMatchTolerance, def | lowopen,
			 "Max fractional deviation of template from tier sigmas", 0.05, 0.);
    parameters.addMember("nThreads", &nThreads, def,
			 "Number of threads to use (-1=all)", -1);
      
    /** At the moment there is nothing random in this code, and no chunking
    parameters.addMember("chunk", &chunk, def | low,
			 "Batch size dispatched to each thread", 100L, 1L);
    parameters.addMember("seed", &seed, def | low,
			 "Random number seed (0 uses time of day)", 0L, 0L);
    **/
  }

  parameters.setDefault();
  
  try {

    int positionalArguments;
    bool badInput = false;
    try {
      // First read the command-line arguments so we know how many positional
      // arguments precede them.
      positionalArguments = parameters.setFromArguments(argc, argv);
    } catch (std::runtime_error &m) {
      // An error here might indicate someone entered "-help" or something
      badInput = true;
      cerr << m.what() << endl;
    }
    if (badInput || positionalArguments!=3) {
      cerr << usage << endl;
      cerr << "#---- Parameter defaults: ----" << endl;
      parameters.dump(cerr);
      exit(1);
    }

    string noisetierFile = argv[1];
    string templateFile = argv[2];
    
    // And now re-read the command-line arguments so they take precedence
    parameters.setFromArguments(argc, argv);

    cout << "#" << stringstuff::taggedCommandLine(argc,argv) << endl;
    parameters.dump(cout);

    /***************************
    Do some more checking of the parameters
    ***************************/
    
    Stopwatch timer;

    // Open the NoiseTier file
    timer.reset();
    timer.start();
    NoiseTierCollection<BC> tiers(noisetierFile,true); // Update existing file.
    timer.stop();
    cerr << "# Read noise tier file in " << timer << endl;

    // Open the template file
    timer.reset();
    timer.start();
    TemplateTable<BC> templates(templateFile);
    timer.stop();
    cerr << "# Read templates file in " << timer << endl;

    // Get our noise tier.
    int useTier = templates.getTier();
    cerr << "# Using noise tier " << useTier << endl;
    // Get pointer to our chosen NoiseTier
    auto ntptr = tiers[useTier];
    // Get the specs for template spacing etc, 
    auto tsTier = ntptr->needsTemplateSpecs();

    /**/cerr << "got specs, wtN " << tsTier.weightN << endl;
    
    // Are templates good enough?
    auto tsTemplate = templates.getTemplateSpecs();
    if (!tsTemplate.goodEnoughFor(tsTier,
				  priorMatchTolerance)) {
      cerr << "ERROR: Template sampling not a sufficient match to tier specs" << endl;
      exit(1);
    }
    
    // Get our sampling from the templates
    double priorSigmaMax= tsTemplate.sigmaMax;
    double priorSigmaStep = tsTemplate.sigmaStep;
    
    {
      // Record this operation in header of noise tier
      std::time_t rawtime;
      struct std::tm* timeinfo;
      char buffer[40];
      std::time (&rawtime);
      timeinfo = std::gmtime(&rawtime);
      strftime(buffer, 40 , "%F %T", timeinfo);
      string t = buffer;
      ostringstream oss;
      oss << t
	  << "Calculating PqrSel from templates " << templateFile;
      ntptr->addHistory(oss.str());
    }


    Pqr<BC> selectionPqr;
    // The Selector classes expect 0's for open flux bounds.
    double fluxMin = max(0.,tsTier.fluxMin);
    double fluxMax = max(0.,tsTier.fluxMax);
    cout << "# ****Selection flux range " << fluxMin << " - " << fluxMax << endl;

    // Configure the calculations
#ifdef _OPENMP
    if(nThreads>0) omp_set_num_threads(nThreads);
#endif

    // Make this ud because Prior interface needs it; selection probability
    // currently does not use it.
    ran::UniformDeviate<double> ud;

    // No need for KdTree for completeness calculation, just make a linear prior
    Prior<BC> prior(fluxMin, fluxMax,
		    ntptr->nominalCov(),
		    ud,
		    true, // Makes this calculate selection
		    1.,   // no noise magnification needed
		    priorSigmaStep, priorSigmaMax,
		    false); // Allow covariance to vary

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
    cout << "# Total template density: " << prior.totalDensity() << endl;
    timer.reset();
    timer.start();
    // Prepare the prior (builds tree, etc.)
    prior.prepare();
    timer.stop();
    cerr << "# prepared in " << timer << endl;

    // Loop over all covariances in the tier
    TargetGalaxy<BC> targ;
    for (int i=0; i<ntptr->nCov(); i++) {
      targ.cov = ntptr->getCov(i);
      // Do the selection integration
      timer.reset();
      timer.start();
      int junk;
      selectionPqr = prior.getPqr(targ, junk, junk);
      timer.stop();
      cerr << "# Selection " << i << " calculated in " << timer << endl;

      // Normalize the PQR to the total area of the templates
      selectionPqr /= templates.sampleArea();

      // Print result and store with targets 
      ntptr->setPqr(selectionPqr, i);
      
    } // End loop over covariance rows
    // The destructors will write info back to the tier table.
  } catch (std::runtime_error& m) {
    quit(m,1);
  }

  exit(0);
}
