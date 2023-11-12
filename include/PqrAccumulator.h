// The PqrAccumulator class is where many templates are compared to a given target
// to execute the integration over template space.
// Also included here is a structure holding the needed info for each template.

#ifndef PQRACCUMULATOR_H
#define PQRACCUMULATOR_H

#include <set>
#include "Std.h"
#include "BfdConfig.h"
#include "Pqr.h"
#include "Moment.h"
#include "Galaxy.h"
#include "Selection.h"
#include "PqrCalculation.h"

namespace bfd {

  // Structure to hold all template information needed in Prior integration
  // Contains all the derivative structures, source id, weight;
  // *AND* another vector of concatenated even & odd moments which
  // *MAY BE IN A DIFFERENT BASIS* and which will be used to build tree
  // or other locating structures.
  template <class CONFIG>
  class TemplateInfo {
  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    typedef typename BC::MXYVector MXYVector;

    typename BC::MDMatrix dm; // Values + derivs of even moments
    typename BC::XYDMatrix dxy; // Values + derivs of odd (xy) moments
    FP nda;  // weight = sky density times xy shift cell area
    long id; // ID of original template before shift/rotation copies

    MXYVector m;  // Concatenated moment vector

    TemplateInfo(): dm(0), dxy(0), nda(1), id(0), m(0) {}
    TemplateInfo(const TemplateGalaxy<BC>& tmpl):
      dm(tmpl.realMDerivs()),
      dxy(tmpl.realXYDerivs()),
      nda(tmpl.nda),
      id(tmpl.id),
      m(FP(0)) {
      // Fill in alternate moment vector with values in deriv structures
      m.subVector(0,BC::MSIZE) = dm.col(BC::P);
      if (!BC::FixCenter)
	m.subVector(BC::MSIZE, BC::MXYSIZE) = dxy.col(BC::P);
    }
    
    // Need interface to use tree
    const MXYVector getM() const {return m;}
    void setM(const MXYVector& _m) {m=_m;}
    
    EIGEN_NEW
    // ??? Could cache dm Cinv dm 2nd deriv matrix here
  };

  // Subclass that accumulates Pqr values for a single target as it is fed templates.
  template <class CONFIG>
  class PqrAccumulator {
  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    PqrAccumulator(const TargetGalaxy<BC>& target,
		   const Selector<BC>* select_,
		   FP sigmaMax_,
		   bool applyWeights_,
		   bool invariantCovariance_);

    void accumulate(const TemplateInfo<BC>* tmpl);
    long nTemplates;  // Number of templates with non-zero contribution
    // Get number of unique id's contributing to integration:
    int nUnique() const {return ids.size();} 
    static int idCeiling;	// Stop counting unique id's above this number
    Pqr<BfdConfig<BC::FixCenter,
      BC::UseConc,
      BC::UseMag,
      BC::Colors,
      false>> total;  // Force double precision in the accumulator Pqr
    EIGEN_NEW
  private:
    const MLike<BC> ml;
    const XYLike<BC> xyl;
    const Selector<BC>* select;
    bool applyWeights;	// True if we are sampling by number and need to apply weights to sums
    bool invariantCovariance;
    bool stillCounting;	// True if we are still looking for unique id's.

    FP selectionScalar;   // The selection factor, if it is independent of template props
    set<int> ids;  // set of the id's used in integration
    
  }; // end of PqrAccumulator class

} // namespace bfd

#endif // PQRACCUMULATOR_H
