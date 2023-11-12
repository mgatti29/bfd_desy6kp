// classesw read target or template galaxy data from
// FITS tables created using momenttable.py.
// *** Index values, column names, header keywords in this file must
// *** be kept in sync with those in the python file!!!
// The Python code always produces a full set of moments.  This code
// will only report the ones in use given the BfdConfig given as CONFIG.
// This code hence does not *write* moments into the table.
// It can

#ifndef MOMENTTABLE_H
#define MOMENTTABLE_H

#include "BfdConfig.h"
#include "Galaxy.h"
#include "Header.h"
#include "FTable.h"
#include "Image.h"
#include "Pqr.h"
#include "FitsImage.h"
#include "FitsTable.h"

namespace bfd {

  // Structure containing specifications for making a template set
  struct TemplateSpecs {
  public:
    // Construct either empty
    TemplateSpecs(): weightN(0), weightSigma(0.),
		     fluxMin(-1.), fluxMax(-1.),
		     sigmaXY(-1.), sigmaFlux(-1.),
		     sigmaMax(-1.), sigmaStep(-1.) {};
    // Or from a header
    TemplateSpecs(const img::Header& h);
    
    // Specify the weight function used
    int weightN;
    double weightSigma;
    // Flux range for selection; <= 0 means unbounded
    double fluxMin;
    double fluxMax;
    // Sigmas for translational copying
    double sigmaXY;
    double sigmaFlux;
    // Spacing/range for sampling; <=0 means no demands yet
    double sigmaMax;
    double sigmaStep;

    bool goodEnoughFor(const TemplateSpecs& rhs,
		       double tolerance=0.05) const {
      // Would this set of template specs suffice for
      // use with a noise tier that wants the rhs specs?
      if (true) {  //***** for debugging mismatches
	cerr << weightN << " vs " << rhs.weightN << endl;
	cerr << weightSigma << " vs " << rhs.weightSigma << endl;
	cerr << fluxMin << " vs " << rhs.fluxMin << endl;
	cerr << fluxMax << " vs " << rhs.fluxMax << endl;
	cerr << sigmaXY << " vs " << rhs.sigmaXY << " " << close(sigmaXY,rhs.sigmaXY,tolerance) << endl;
	cerr << sigmaFlux << " vs " << rhs.sigmaFlux << " "
	     << close(sigmaFlux,rhs.sigmaFlux,tolerance) << endl;
	cerr << sigmaMax << " vs " << rhs.sigmaMax << endl;
	cerr << sigmaStep << " vs " << rhs.sigmaStep << endl;
      } 
      return weightN==rhs.weightN 
	// The weight functions match
	&& close(weightSigma,rhs.weightSigma) 
	// If rhs has a fluxMin, we must be below it, if not we also have no min
	&& (rhs.fluxMin>0 ? fluxMin <= rhs.fluxMin : fluxMin<=0) 
	// If rhs has a fluxMax, ours must be higher; if not, also have no max here
	&& (rhs.fluxMax>0 ? fluxMax <= 0 || fluxMax>=rhs.fluxMax: fluxMax <= 0)
	// sigmaXY and sigmaFlux close, if noise tier has them
	&& (rhs.sigmaXY<=0. || close(sigmaXY,rhs.sigmaXY,tolerance))
	&& (rhs.sigmaFlux<=0. || close(sigmaFlux,rhs.sigmaFlux,tolerance))
	// sigmaMax goes at least as far, and step no larger than spec'ed, if at all
	&& (rhs.sigmaMax<= 0 || sigmaMax >= rhs.sigmaMax)
	&& (rhs.sigmaStep<=0 || sigmaStep <= rhs.sigmaStep);
    }

    // Add/modify keywords in header to encode this information
    void addToHeader(img::Header& h) const;

  private:
    static bool close(double a, double b, double tol=1e-4) {
      // Helper function for goodEnoughFor
      return abs(b/a-1.) < tol;
    }

  };
    
	
  template <class CONFIG>
  class TargetTable {
  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    // Construct with update=true to automatically write back changes
    // upon destruction of this class.
    TargetTable(string fitsname, bool update=false);
    ~TargetTable();
    // No copying
    void operator=(const TargetTable<CONFIG>& rhs) =delete;
    TargetTable(const TargetTable<CONFIG>& rhs) = delete;

    long size() const {return tab.nrows();}

    // Return the nth target galaxy
    TargetGalaxy<BC> getTarget(long index) const;

    // Set Pqr for a row (creates column if needed)
    void setPqr(const Pqr<BC>& pqr, long index);
    // Or add to current value
    void addPqr(const Pqr<BC>& pqr, long index);
    // Or retrieve
    Pqr<BC> getPqr(long index) const;
    // Add appropriate column to table (with zeros) if needed:
    void checkPqrColumn();
    // Add appropriate column for nUnique to table, if needed
    void checkNUniqueColumn();
    // Add appropriate column (non-)detection area to table, if needed
    void checkAreaColumn();
    // Get/set nUnique for a row (creates column if needed)
    void setNUnique(int nUnique, long index);
    int getNUnique(long index) const;
    // Get/set the area assigned to a (non)-detection
    void setArea(double dA, long index);
    double getArea(long index) const;
    // Get noise Tier for a row
    int getTier(long index) const;
    // Retrieve covariance for selected row
    MomentCovariance<BC> getCov(long index) const;

    // Report whether galaxies use nominal tier covariance matrix or all have their own
    bool hasSharedCov() const {return sharedCov;}

    // Various diagnostic information
    double weightN() const {return wtN;}     // Weight function with which targets were measured
    double weightSigma() const {return wtSigma;}

    // Add history to the catalog header
    void addHistory(string h);
    
    // Write to a FITS file - will overwrite any existing file
    void save(string fitsname) const;
    
    // These indices must match what's in momentcalc.Moment:
    const static int M0 = 0;
    const static int MR = 1;
    const static int M1 = 2;
    const static int M2 = 3;
    const static int MC = 4;
    const static int NM = 5; // Not using xy moments since they're 0 for targets

    static vector<int> iMap;  // Gives the Python index for each even moment in use here
    static void buildIMap();  // create the python-index lookup table

    // Build a MomentCovariance from the packed format
    static MomentCovariance<BC> unpackCov(vector<float> cov);

  private:
    bool updateDisk;       // True to mirror changes to the input file
    bool sharedCov;        // True if table does not have columns for individual cov's
    bool hasTierColumn;    // Does table have a column for noise tiers?
    double wtN;
    double wtSigma;
    img::FTable tab;

    // The FITS extension, which must be kept
    // open if we are going to update it.
    FITS::FitsTable* fitsTabPtr;
    
  };

  //*****************************************************************
  // Class to represent a noise tier for targets.
  // It's a wrapper around a binary FITS table.
  template <class CONFIG>
  class NoiseTier {
  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    NoiseTier(img::FTable ft): tab(ft) {}
    ~NoiseTier() {};
    // No copying
    void operator=(const NoiseTier<CONFIG>& rhs) =delete;
    NoiseTier(const NoiseTier<CONFIG>& rhs) = delete;

    // Get its id number
    int getID() const;
    
    // Query number of covariances in the table
    int nCov() const {return tab.nrows();}

    // Set Pqr for a row of covariances
    void setPqr(const Pqr<BC>& pqr, int index);
    // Or retrieve
    Pqr<BC> getPqr(int index) const;
    // Add appropriate column to table (with zeros) if needed:
    void checkPqrColumn();

    // Aquire a full covariance from a given row
    MomentCovariance<BC> getCov(int index=0) const;
    // Aquire nominal covariance (which is row zero by convention)
    MomentCovariance<BC> nominalCov() const {
      return getCov(0);
    }

    // What the noise tier wants from its templates
    TemplateSpecs needsTemplateSpecs() const {
      return TemplateSpecs(*tab.header());
    }
    void setTemplateSpecs(const TemplateSpecs& ts) {
      ts.addToHeader(*tab.header());
    }

    // Add history to the catalog header
    void addHistory(string h);
    
  private:
    img::FTable tab;   // Recall that these tables are link-counted
  };

  template <class CONFIG>
  class NoiseTierCollection {
  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    // Set update=true in constructor to write changes back
    // to file upon destruction
    NoiseTierCollection(string filename, bool update=false);
    ~NoiseTierCollection();
    // No copying
    void operator=(const NoiseTierCollection& rhs) =delete;
    NoiseTierCollection(const NoiseTierCollection& rhs) =delete;

    // Return vector of all existing noise tier id's.
    vector<int> allTiers() const {
      vector<int> out;
      for (const auto& pr : tiers) {
	out.push_back(pr.first);
      }
      return out;
    }

    bool hasTier(int id) const {
      return tiers.count(id);
    }
    const NoiseTier<BC>* operator[](int id) const {
      return tiers.at(id);
    }
    NoiseTier<BC>* operator[](int id) {
      return tiers.at(id);
    }

  private:
    std::map<int,NoiseTier<BC>*> tiers;
    // The FITS pointers are saved if we want to update on destruction
    list<FITS::FitsTable*> savedFITS;

  };


  template <class CONFIG>
  class TemplateTable {
  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    typedef TemplateGalaxy<BC> TG;
    TemplateTable(string fitsname);

    long size() const {return tab.nrows();}

    // Return the nth target galaxy
    TemplateGalaxy<BC> getTemplate(long index) const;

    // Various diagnostic information

    TemplateSpecs getTemplateSpecs() {
      return TemplateSpecs(*tab.header());
    }
    
    // Sky area (or number of stamps) which templates represent
    double sampleArea() const {return effArea;}

    // Get tier number this was made for
    int getTier() const;
    
    // Indices that must match what's in momentcalc.Moment:
    const static int M0 = 0;
    const static int MR = 1;
    const static int M1 = 2;
    const static int M2 = 3;
    const static int MC = 4;
    const static int MX = 5;
    const static int MY = 6;
    const static int NM = 7;

  private:
    img::FTable tab;
    double effArea;
    
    // Get cell from column and save in TMV vector
    void readM(linalg::Vector<FP>& out, string colName, long irow) const;
    // Turn vector z of values for all real moments into row for of complex-valued moments
    void saveM(linalg::Vector<std::complex<FP>>& z,
	       typename TG::DerivMatrix& mderiv,
	       typename TG::DerivMatrix& xyderiv,
	       int irow) const;
  };
    
} // namespace bfd
#endif // MOMENTTABLE_H
