// Classes that produce values and derivatives wrt lensing of commonly used
// variable combinations.

#ifndef PQRCALCULATION_H
#define PQRCALCULATION_H

#include "BfdConfig.h"
#include "Moment.h"
#include "Pqr.h"
#include "Galaxy.h"

namespace bfd {

  /******************************************************
  The main calculation of the method: give 
  the multivariate Gaussian likelihood of M moments 
  of the target arising from the template.
  The pqr call gives derivs of this likelihood wrt lensing.
  The probabilities returned here do *not* have the 
  normalization constant for the Gaussian.
  ******************************************************/
  template <class CONFIG>
  class MLike {
  public:
    typedef CONFIG BC;
    typedef typename CONFIG::FP FP;
    MLike(const TargetGalaxy<BC>& targ,
	  FP sigmaMax=6.5);
    // As a speedup, the derivative operator can abort
    // if the chisq between target and template is
    // >max_chisq, which will be signalled by having
    // out[BC::P]<0 in the output Pqr vector.  The
    // chisqStart value (i.e. coming from xy moments)
    // is added to the chisq calculated here in this decision.

    FP operator()(const typename BC::MVector& mG) const;
    Pqr<BC> operator()(const typename BC::MDMatrix& dmG,
		       FP chisqStart=0.) const;

    // Allow reset of target moments, keeping old
    // covariance matrix inverse:
    void setTargetMoments(const typename BC::MVector& m) {
      mTarget = m;
    }
  private:
    typedef typename BC::MVector MVector;
    typedef typename BC::MMatrix MMatrix;
    typename BC::MVector mTarget;
    typename BC::MMatrix invcov;
    FP sigmaMaxSquared;  // abort templates with chi2 above this
  };


  /******************************************************
  The following class is used whenever centroid moments are being nulled.
  Calculates the multivariate Gaussian likelihood of X and Y moments being
  zero given the template's moments are xy.
  The pqr call gives derivs of this likelihood wrt lensing.
  ******************************************************/
  template <class CONFIG>
  class XYLike {
  public:
    typedef CONFIG BC;
    typedef typename CONFIG::FP FP;
    XYLike(const TargetGalaxy<BC>& targ); // Only uses xy covariance
    FP operator()(typename BC::XYVector xy) const;
    Pqr<BC> operator()(const typename BC::XYDMatrix& dxy) const;
    // This returns just the chisq in the exponential:
    FP chisq(typename BC::XYVector xy) const;
  private:
    FP normLxy; // Components of the likelihood L(mx,my)
    FP invCxx;
    FP invCxy;
    FP invCyy;
  };

  /******************************************************
  Likewise this gives the determinant of the Jacobian dM[xy]/d[xy],
  and its derivatives under lensing.  This Jacobian is simply
  expressed in terms of the quadratic moments, fortunately.
  There is no persistent information in this class, zero size.
  Despite the name it's a function of M moments, not XY
  ******************************************************/
  template <class CONFIG>
  class XYJacobian {
  public:
    typedef CONFIG BC;
    typedef typename CONFIG::FP FP;
    FP operator()(const typename BC::MVector& m) const;
    Pqr<BC> operator()(const typename BC::MDMatrix& dm) const;
  };
} // end namespace bfd

#endif //PQRCALCULATION_H
