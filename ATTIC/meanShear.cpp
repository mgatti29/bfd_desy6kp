// Estimate mean shear of a target catalog

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
#include "Pset.h"
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
  "meanShear: estimate mean shear and uncertainty of a FITS catalog of targets.\n"
  "   Assumes that all integrations against templates are completed already.\n"
  "   *targetFile* is the path to a target galaxy catalog in FITS standard fmt.\n"
  "   If *stampSelection*=true, a PQR factor is calculated for the probability\n"
  "     of non-selection using statistics appropriate to a postage-stamp simulation.\n"
  "     The target file must contain an TIERLOST count of undetected stamps\n"
  "     in each noise tier extension being used.\n"
  "   If *poissonSelection*=true, a PQR factor is calculated for the probability\n"
  "     of non-selection using statistics appropriate to Poisson sampling.  The\n"
  "     target file must contain an TIERAREA measure of the sky area covered\n"
  "     by each noise tier.\n"
  "   *skipTiers* is comma-separated list of tiers to exclude from estimate.\n"
  "   The stdout of the program is the total log posterior for lensing assuming postage-stamp\n"
  "     statistics, and including deselection effects if doSelection=true.\n"
  "Usage: meanShear [parameter file] [-parameter [=] value...]\n"
  "       Parameter files read in order of command line, then command-line parameters,\n"
  "       later values override earlier.\n"
  "stdin: none\n"
  "Run with -help to get parameter list\n";

int main(int argc,
	 char *argv[])
{
  string targetFile;
  bool stampSelection;
  bool poissonSelection;
  string skipTiers;
  int minUniqueTemplates;

  Pset parameters;
  {
    const int def=PsetMember::hasDefault;
    const int low=PsetMember::hasLowerBound;
    const int up=PsetMember::hasUpperBound;
    const int lowopen = low | PsetMember::openLowerBound;
    const int upopen = up | PsetMember::openUpperBound;

    parameters.addMember("targetFile",&targetFile, def,
			 "Input target moment table filename", "");
    parameters.addMember("stampSelection",&stampSelection, def,
			 "Include selection term for postage-stamp statistics?", false);
    parameters.addMember("poissonSelection",&poissonSelection, def,
			 "Include selection term for poisson statistics?", false);
    parameters.addMember("skipTiers",&skipTiers, def,
			 "Which noise tiers to exclude", " ");
    parameters.addMember("minUniqueTemplates",&minUniqueTemplates, def | low,
			 "Min number of templates used for valid target", 1, 1);
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
    
    if (stampSelection && poissonSelection) {
      cerr << "Cannot choose both stampSelection and poissonSelection" << endl;
      exit(1);
    }
    
    // Do we have files?
    if (targetFile.empty()) {
      cerr << "ERROR: Need to specify targetFile" << endl;
      exit(1);
    }

    // Open the target file
    TargetTable<BC> targets(targetFile);

    // Put the tier numbers into a set for fast checking
    std::set<int> tierSet;
    {
      auto tmp = targets.allTiers();
      tierSet.insert(tmp.begin(), tmp.end());
    }

    // Remove excluded tiers
    for (string s : stringstuff::split(skipTiers,',')) {
      string t = s;
      stringstuff::stripWhite(t);
      if (t.empty())
	continue;
      // Check that it's entirely a positive integer
      if (t.find_first_not_of( "0123456789" ) != std::string::npos) {
	cerr << "Invalid skipTiers element <" << t << ">" << endl;
	exit(1);
      }
      int drop = atoi(t.c_str());
      tierSet.erase(drop);
    }

    // Collect the lost-stamp count of tiers if needed
    std::map<int,long> lostCount;
    if (stampSelection) {
      for (int tier: tierSet) {
	lostCount[tier] = targets.getTierLost(tier);
	if (lostCount[tier] < 0) {
	  cerr << "Missing needed TIERLOST from tier " << tier << endl;
	  exit(1);
	}
      }
    }

    // Collect the survey area of tiers if needed
    std::map<int,double> tierArea;
    if (poissonSelection) {
      for (int tier: tierSet) {
	tierArea[tier] = targets.getTierArea(tier);
	if (tierArea[tier] < 0.) {
	  cerr << "Missing needed TIERAREA from tier " << tier << endl;
	  exit(1);
	}
      }
    }

    // Now pass through all targets, summing up log(prob)
    Pqr<BCD> accumulator;
    long nUsed = 0;
    for (long i=0; i<targets.size(); i++) {
      // Use only requested tiers
      int tier = targets.getTier(i);
      if (tierSet.count(tier)<=0)
	continue;

      auto pqr = targets.getPqr(i);
      if (pqr[BC::P] > 0. && targets.getNUnique(i) >= minUniqueTemplates) {
	++nUsed;
	accumulator += pqr.neglog();
      } else if (stampSelection) {
	// Treat as non-detection
	lostCount[tier] = lostCount[tier]+1;
      }
    }

    if (stampSelection) {
      // Add the deselection terms to PQR for each tier
      for (auto pr : lostCount) {
	int tier = pr.first;
	long nDeselect = pr.second;
	if (nDeselect > 0) {
	  auto deselectionPqr = targets.getPqrSelect(tier);
	  // deselection prob is 1-selection
	  deselectionPqr *= FP(-1.);  
	  deselectionPqr[BC::P] += FP(1);
	  if (deselectionPqr[BC::P] <= 0.) {
	    cout << "# WARNING: deselection galaxies occurred with zero probability at tier "
		 << tier
		 << endl;
	  } else {
	    Pqr<BC> tmp(deselectionPqr.neglog());
	    tmp *= (FP) nDeselect;
	    accumulator += tmp;
	  }
	}
      }
    }

    if (poissonSelection) {
      // Add deselection terms for Poisson sampling
      for (auto pr : lostCount) {
	int tier = pr.first;
	double area = pr.second;
	auto deselectionPqr = targets.getPqrSelect(tier);
	// Check this ???
	deselectionPqr *= FP(-area);  
	accumulator += deselectionPqr;
      }
    }

    BCD::QVector gMean;
    BCD::RMatrix gCov;
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
