// Calculate the selection PQR for a given noise tier of a target table
// given a template table, in standard FITS table formats.

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

bool
checkTemplateNoiseMatch(const TemplateTable<BC>& templates,
			const MomentCovariance<BC>& mcov,
			double sigmaSlop) {
  // Check to see whether the template table was built with sigma
  // values close to the given covariance matrix's.
  return abs(templates.sigmaFlux()/sqrt(mcov.m(BC::MF,BC::MF))-1.) <= sigmaSlop && 
    abs(templates.sigmaXY()/sqrt(mcov.xy(BC::MX,BC::MX))-1.) <= sigmaSlop &&
    abs(templates.sigmaXY()/sqrt(mcov.xy(BC::MY,BC::MY))-1.) <= sigmaSlop;
}


const string usage =
  "calculateSelection: Calculate the selection PQR for a given noise tier of a target table\n"
  "   given a template table, in standard FITS table formats.\n"
  "   The *selectSN* parameter is a comma-separated list of values defining a series of\n"
  "     intervals in flux S/N into which targets will be divided.  Zero means no bound,\n"
  "     and the default is that no selection cut is made. A single number will be taken\n"
  "	as a lower bound with no upper.\n"
  "   The *selectMF* parameter alternatively defines these intervals as absolute flux.\n"
  "   The *useTiers* parameter is number of the noise tier of targets to process\n"
  "     in this run.  By default (-1), the first tier that matches the template file will be done.\n"
  "   *priorMatchTolerance* gives the maximum fractional deviation that flux or centroid sigma\n"
  "     used in building the templates can have from the nominal target values.\n"
  "   If *outfile* parameter is given, the updated target table is written to it.\n"
  "   The *priorSigmaStep, priorSigmaMax* values are taken from templatefile unless overridden by\n"
  "     specification of positive values in the parameters.\n\n"
  "   The calculated selection PQR is reported to stdout and (over)written into the header\n"
  "   of the noise tier extension.\n"								
  "Usage: calculateSelection [parameter file] [-parameter [=] value...]\n"
  "       Parameter files read in order of command line, then command-line parameters,\n"
  "       later values override earlier.\n"
  "stdin: none\n"
  "     Run with -help to get parameter list\n"
  "stdout: The derived selection PQR.";

int main(int argc,
	 char *argv[])
{
  double priorSigmaMax;
  double priorSigmaStep;
  double priorSigmaBuffer;
  double priorMatchTolerance;

  int nThreads;
  long chunk;
  long seed;
  bool sampleWeights;

  string targetFile;
  string templateFile;
  
  string selectSN;
  string selectMF;
  int useTier;

  Pset parameters;
  {
    const int def=PsetMember::hasDefault;
    const int low=PsetMember::hasLowerBound;
    const int up=PsetMember::hasUpperBound;
    const int lowopen = low | PsetMember::openLowerBound;
    const int upopen = up | PsetMember::openUpperBound;

    parameters.addMember("targetFile",&targetFile, def,
			 "Input target moment table filename", "");
    parameters.addMember("templateFile",&templateFile, def,
			 "Input template moment table filename", "");
    parameters.addMember("selectSN",&selectSN, def,
			 "S/N bin divisions", "");
    parameters.addMember("selectMF",&selectMF, def,
			 "flux moment bin divisions", "");
    parameters.addMember("useTier",&useTier, def,
			 "Which noise tier to process", -1);

    parameters.addMemberNoValue("PRIOR:",0,
				"Characteristics of the sampled prior");
    parameters.addMember("priorSigmaMax",&priorSigmaMax, def,
			 "Maximum sigma range when sampling for prior", 0.);
    parameters.addMember("priorSigmaStep",&priorSigmaStep, def,
			 "Step size when sampling for prior", 0.);
    parameters.addMember("priorSigmaBuffer",&priorSigmaBuffer, def | low,
			 "Buffer width of KdTreePrior (in sigma)", 1., 0.);
    parameters.addMember("priorMatchTolerance",&priorMatchTolerance, def | lowopen,
			 "Max allowed deviation of template from target sigmas", 0.05, 0.);
      
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
    if (targetFile.empty() || templateFile.empty()) {
      cerr << "ERROR: Need to specify targetFile and templateFile parameters" << endl;
      exit(1);
    }

    Stopwatch timer;

    // Open the target file
    timer.reset();
    timer.start();
    TargetTable<BC> targets(targetFile,true); // Update existing file.
    timer.stop();
    cerr << "# Read targets file in " << timer << endl;

    // Open the template file
    timer.reset();
    timer.start();
    TemplateTable<BC> templates(templateFile);
    timer.stop();
    cerr << "# Read templates file in " << timer << endl;

    // Check agreement of weight scales btwn target & templates
    if (targets.weightN() != templates.weightN() ||
	abs(targets.weightSigma()/templates.weightSigma()-1) > 0.0001) {
      cerr << "ERROR: Mismatch between target and template weight functions" << endl;
      cerr << "N: " << targets.weightN() << " vs " << templates.weightN() << endl;
      cerr << "Sigma: " << targets.weightSigma() << " vs " << templates.weightSigma() << endl;
      exit(1);
    }

    // Check any overrides on sigmaStep, sigmaMax
    if (priorSigmaStep <= 0.) {
      priorSigmaStep = templates.sigmaStep();
    } else if (priorSigmaStep < templates.sigmaStep()) {
	cerr << "ERROR: Specified priorSigmaStep is smaller than used for templates" << endl;
	exit(1);
    }

    if (priorSigmaMax <= 0.) {
      priorSigmaMax = templates.sigmaMax();
    } else if (priorSigmaMax > templates.sigmaMax()) {
	cerr << "ERROR: Specified priorSigmaMax is larger than used for templates" << endl;
	exit(1);
    }

    // Acquire a nominal covariance matrix from one of the tiers
    int noiseTier = -1;
    MomentCovariance<BC> mcov;
    if (useTier < 0) {
      // We have been invited to find a noise tier that matches the templates
      for (int tier : targets.allTiers()) {
	mcov = targets.getNominalCov(tier);
	if (checkTemplateNoiseMatch(templates,
				    mcov,
				    priorMatchTolerance)) {
	    noiseTier = tier;
	    break;
	}
      }
      if (noiseTier < 0) {
	// No match.
	cerr << "ERROR: Template does not match any of targets' noise tiers" << endl;
	exit(1);
      }
    } else {
      noiseTier = useTier;
      mcov = targets.getNominalCov(noiseTier);
      if (!checkTemplateNoiseMatch(templates,
				   mcov,
				   priorMatchTolerance)) {
	// requested tier is a bad match, quit
	cerr << "ERROR: Template and target flux errors are too different" << endl;
	cerr << "Flux:  " << templates.sigmaFlux() << " vs " << sqrt(mcov.m(BC::MF,BC::MF)) << endl;
	cerr << "X:     " << templates.sigmaXY() << " vs " << sqrt(mcov.xy(BC::MX,BC::MX)) << endl;
	cerr << "Y:     " << templates.sigmaXY() << " vs " << sqrt(mcov.xy(BC::MY,BC::MY)) << endl;
	exit(1);
      }
    }
    cerr << "# Using covariance tier " << noiseTier << " as nominal" << endl;
    {
      std::time_t rawtime;
      struct std::tm* timeinfo;
      char buffer[40];
      std::time (&rawtime);
      timeinfo = std::gmtime(&rawtime);
      strftime(buffer, 40 , "%F %T", timeinfo);
      string t = buffer;
      ostringstream oss;
      oss << t
	  << "Calculating PqrSelect for tier " << noiseTier << " from templates " << templateFile;
      targets.addHistory(oss.str());
    }

    // Next parse the requested S/N or flux ranges.  Only one allowed
    if ( !selectSN.empty() && !selectMF.empty()) {
      cerr << "ERROR: cannot specify both S/N and flux moment selection limits" << endl;
      exit(1);
    }

    string selectString = selectSN.empty() ? selectMF : selectSN;
    
    // Pull apart the selection ranges and check them
    vector<double> vFlux;
    for (auto sn : stringstuff::split(selectString, ',')) {
      vFlux.push_back(atof(sn.c_str()));
    }
    if (vFlux.size()==0) {
      // Nothing entered, use no cuts
      vFlux.push_back(0.);
    }
    if (vFlux.size()==1) {
      // Only one number entered means no upper limit
      vFlux.push_back(0.);
    }
    // Check for increasing ranges; except that the last value could
    // be zero to indicate unbounded upper range
    int nRanges = vFlux.size()-1;
    if (vFlux[0] <0) vFlux[0] = 0.;
    if (vFlux.back() <= 0) {
      vFlux[nRanges] = 0.;
    } else if (vFlux[nRanges] <= vFlux[nRanges-1]) {
      cerr << "ERROR: selectSN or selectMF values are not increasing" << endl;
      exit(1);
    }
    for (int i=0; i<nRanges-1; i++) {
      if (vFlux[i+1] <= vFlux[i]) {
	cerr << "ERROR: selectSN values are not increasing" << endl;
	exit(1);
      }
    }

    // Rescale S/N to flux if that's what we read
    double sigmaF = sqrt(mcov.m(BC::MF,BC::MF));
    if (!selectSN.empty()) {
      for (int i=0; i<vFlux.size(); i++)
	vFlux[i] *= sigmaF;
    }
    
    // Check agreement of template fluxMin against chosen values
    if (templates.snMin()*templates.sigmaFlux() > vFlux[0]) {
      cerr << "ERROR: Template fluxMin is higher than selection cutoff of targets" << endl;
      exit(1);
    }

    Pqr<BC> selectionPqr;
    double fluxMin = vFlux.front();
    double fluxMax = vFlux.back();
    if (vFlux.front() <= 0 && vFlux.back() <= 0) {
      // No selection is being done.  Save zeros.
      cout << "# ****No selection cuts are being made" << endl;
      targets.setPqrSelect(noiseTier,selectionPqr);
    } else {
      // Configure the calculations
#ifdef _OPENMP
      if(nThreads>0) omp_set_num_threads(nThreads);
#endif

      ran::UniformDeviate<double> ud;
      if (seed > 0) ud.seed(seed);

      //****************
      // Calculate the selection PQR
      cout << "# ****Selection flux range " << fluxMin << " - " << fluxMax << endl;

      // No need for KdTree for completeness calculation, just make a linear prior
      Prior<BC> prior(fluxMin, fluxMax,
		      mcov,
		      ud,
		      true, // Makes this calculate selection
		      1.,   // no noise magnification needed
		      priorSigmaStep, priorSigmaMax,
		      true); // Invariant covariance

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

      // Do the selection integration
      timer.reset();
      timer.start();
      int junk;
      selectionPqr = prior.getPqr(TargetGalaxy<BC>(), junk, junk);
      timer.stop();
      cerr << "# Selection calculated in " << timer << endl;
    }

    // Normalize the PQR to the total area of the templates
    selectionPqr /= templates.sampleArea();
    
    // Print result and store with targets 
    cout << "# Selection PQR: " << selectionPqr << endl;
    targets.setPqrSelect(noiseTier, selectionPqr);
    // Note that destructor will write table back to disk.
    
  } catch (std::runtime_error& m) {
    quit(m,1);
  }

  exit(0);
}
