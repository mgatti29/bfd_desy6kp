// Classes representing priors of moments, and their derivatives w.r.t. shear.
// It is in the Prior::getPqr method that the big Bayesian integration occurs.

#ifndef PRIOR_H
#define PRIOR_H

#include <set>
#include "Std.h"
#include "Random.h"
#include "BfdConfig.h"
#include "Pqr.h"
#include "Moment.h"
#include "Galaxy.h"
#include "Selection.h"
#include "Distributions.h"
#include "PqrCalculation.h"
#include "PqrAccumulator.h"

namespace bfd {


  // This is a prior defined by a set of samples of Moments.
  // Implements the integral with brute-force sum over all templates.
  // Subsampling is implemented in a derived class.

  // fluxmin, fluxmax are bounds enforced for target galaxies to be used.
  //   Selection is made before any noise is added to moments.

  // Nominal MomentCovariance for the targets must be given at construction,
  // and targets should not deviate from this by too much for proper integration.

  // Set invariantCovariance=true on construction to gain some efficiencies if
  // covariance matrix is the same for all targets.  Note that these efficiencies
  // are only gained if we are selectionOnly or if cov matrix is isotropic.

  // Set selectionOnly=true on construction to have this prior built to calculate
  // the total selection probabilities P(s) given a target covariance.
  // Calculates P(M_targ,s) otherwise.

  // Set fixedNoise=true if the galaxies have been centroided with the noise
  // realization on moments being independent of position.  Will alter some formulae.
  // This only happens in some simulation shortcuts.
  // 
  // All target galaxies will be rotated to E2=0 when integrated against prior.
  // Assumes that target galaxies have xy moments=0 and E2=0.
  // Rotated and flipped (but not translated) copies of TemplateGalaxies are
  // added to the prior.
  // sigmaStep determines density of copies, and translations
  // must have already been executed with this density.
  // Template replication is truncated whenever they become sufficiently far 
  //  (>sigmaCutoff times the nominal covariance) from any target galaxy meeting the
  //  MX=MY=M2=0 criteria and the flux selection.

  // noiseFactor>1. on construction will add rotationally symmetric noise to even moments
  //  to augment covariance by this factor (effect is to have
  //  a bigger smoothing kernel for the templates).  

  // Uniform deviate is needed to randomize copy sampling and any other processes.

  // The prepare() method must be called before any targets can be integrated.
  template<class CONFIG>
  class Prior {

  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;

    Prior(FP fluxMin_,
	  FP fluxMax_,
	  const MomentCovariance<BC>& nominalCov,
	  ran::UniformDeviate<double>& ud_,
	  bool selectionOnly_ = false,
	  FP noiseFactor_ = 1.,
	  FP sigmaStep_ = 1.,
	  FP sigmaCutoff_ = 6.5,
	  bool invariantCovariance_ = false,
	  bool fixedNoise_ = false);
    virtual ~Prior() {
      if (nominalSelector) delete nominalSelector;
      if (addedNoiseGenerator) delete addedNoiseGenerator;
      if (gd) delete gd;
      for (auto p : templatePtrs) delete p; // delete TemplateInfo
    }

    // Add a template to the prior. Default is to include flipped as
    // well as rotated copies of it.
    void addTemplate(const TemplateGalaxy<BC>& gal,
		     bool flip=true);

    // Routine that should be called after all template galaxies are added,
    // and before any integrations are done, e.g. to build data structures if needed.
    virtual void prepare();

    // For selectionOnly=false:
    // Returns value and shear derivative of P(M,s) of the target galaxy.
    // Also returns total number of Templates integrated, and number of unique
    // source id's.  Value at BC::P < 0 indicates non-selection.
    // For selectionOnly=true:
    // Returns value and shear derivative of P(s), total selection density
    virtual Pqr<BC> getPqr(const TargetGalaxy<BC> &gal,
			   int& nTemplates,
			   int& nUnique) const;

    int getNTemplates() const {
      return (selectionOnly && invariantCovariance) ? selectionTemplateCounter : templatePtrs.size();}
    double totalDensity() const {return ndaTotal;}

    // Stop counting unique template ID's above this number:
    void setUniqueCeiling(int ceiling);

    EIGEN_NEW
  protected:
    FP fluxMin;
    FP fluxMax;
    FP ndaTotal;  // Total weight/density
    vector<TemplateInfo<BC>* > templatePtrs;  // Pointers to the templates

    const Selector<BC>* nominalSelector;        // Selector with nominal covariance
    // Accumulate selection for nominal covariance
    // in a Pqr that is forced to double precision
    Pqr<BfdConfig<BC::FixCenter,
      BC::UseConc,
      BC::UseMag,
      BC::Colors,
      false>> nominalSelectionPqr;  
    
    // Expected moment covariance matrix to use when building the prior:
    MomentCovariance<BC> nominalCovariance;
    // This is the nominal covariance plus any added noise covariance:
    MomentCovariance<BC> nominalTotalCovariance;

    FP noiseFactor;
    typename BC::MMatrix addedCovariance;
    MultiGauss<FP>* addedNoiseGenerator;
    ran::UniformDeviate<double>& ud;
    ran::GaussianDeviate<double>* gd;

    // Function to add the extra noise to a set of moments
    void addNoiseTo(TargetGalaxy<BC>& m) const;
  
    // Set to exploit speedups when all galaxies have common covariance:
    bool invariantCovariance;

    // A counter for templates used if invariantCovariance && selectionOnly.
    long selectionTemplateCounter;

    // Remove components of covariance matrix that vary under rotation
    static void isotropizeCovariance(MomentCovariance<BC> cov); 
    // Shut off invariantCovariance if the input cov matrix given as argument
    // is different from the isotropized internal one.
    void checkInvariantCovariance(const MomentCovariance<BC>& covIn); 

    bool fixedNoise; // True if moment noise is constant when origin is shifted.
    bool isPrepared;	// Prep needs to be done just once, and no templates added afterwards

    bool selectionOnly; // True if prior will only be used for selection probability

    FP sigmaCutoff; // Allowed to truncate integral beyond this sigma with nominalCovariance
    FP sigmaStep;   // Sample rotations/translations at this density

    // Do the bookkeeping for added noise:
    void setupAddedNoise();

    // Add a single template galaxy to the sampled prior.
    // If flip=true; also add version of template reflected about x axis
    void addSingleTemplate(const TemplateGalaxy<BC> &gal,
			   bool flip=false);


    // Routines to create the proper Selection class for current circumstances of calculation
    // selectionOnly=false will avoid choice of added-noise Selectors.
    const Selector<BC>* chooseSelector(const TargetGalaxy<BC>& targ,
				       const TargetGalaxy<BC>& targAdded) const; 

  }; // end of Prior class declaration

} // namespace bfd

#endif //PRIOR_H
