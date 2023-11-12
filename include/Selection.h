// Subroutines of use in calculating the selection-function contributions to
// the BFD sums over prior
#ifndef SELECTION_H
#define SELECTION_H

#include <cmath>
#include <iostream>
#include "BfdConfig.h"
#include "Pqr.h"
#include "Moment.h"
#include "PqrCalculation.h"
#include "Galaxy.h"

namespace bfd {

  template<class CONFIG>
  class Selector {
    // Interface class to selection functions.
    // The base class implements the most common case of no added noise.
  public:
    typedef CONFIG BC;
    typedef typename CONFIG::FP FP;
    Selector(const FP fluxmin_,
	     const FP fluxmax_,
	     const TargetGalaxy<BC>& targ);

    virtual bool select(const Moment<BC>& mobs) const;
    // Determine whether an observed galaxy is selected
    // Notice that fluxmin=0 or fluxmax=0 means no selection at that limit.
      
    virtual FP prob(const Moment<BC>& mG) const;
    // Returns P(mobs, s, d | mG), the probability of detecting the galaxy at this
    // position AND passing the selection cut, AND obtaining the observed
    // moments, if the underlying galaxy has mG.
    // The value returned here should be multiplied by the noise likelihood
    // L(mobs - mG), and by dA if centroid is free.
    // It is assumed that the observed moments have already passed selection.

    virtual Pqr<BC> probPqr(const typename BC::MDMatrix& dmG) const;
    // Returns the same probability factor as above, plus derivs under lensing

    virtual bool probDependsOnG() const;
    // Return true if the value returned by prob() depends on galaxy properties at all
    
    virtual FP detect(const Moment<BC>& mG) const;
    // Return P(s,d | mG), the total probability of detection and selection
    // of galaxy with true moments mG at this location.
      
    virtual Pqr<BC> detectPqr(const typename BC::MDMatrix& dmG) const;
    // As above, giving derivatives with respect to lensing of the true galaxy

    EIGEN_NEW
  protected:
    const FP fluxmin;
    const FP fluxmax;
    const XYJacobian<BC> jacobian;  // Calculator for Jacobian determinant
    // cached intermediate quantities:
    FP invsigF;               // 1./sqrt(C_ff) for detection prob
    FP jObs;                  // Observed M_xy Jacobian derivative
    typename BC::MVector A;   // 2*B*C_Mf term for detection prob
    FP b;                     // C_fM * B * C_Mf term for detection prob
  };

  
  /******************************************************
  Next is for case when noise has been added *after* selection.
  Note that select() method only makes sense if called with moments before
  extra noise is added.
  The detection probabilities for AddedNoise cases call the
  base class versions w/o added noise, so do not disturb their values.
  ******************************************************/


  template<class CONFIG>
  class AddNoiseSelector: public Selector<CONFIG> {
  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    typedef typename BC::MVector MVector;
    typedef typename BC::MMatrix MMatrix;
    // In the constructor, the TargetGalaxy is needed for both before
    // and after the noise has been added.
    // We won't actually use the moments, just the covariances, in Selectors.
    // Both must have been rotated to the same coordinate basis.
    AddNoiseSelector(const FP fluxmin_,
		     const FP fluxmax_,
		     const TargetGalaxy<BC>& targ,
		     const TargetGalaxy<BC>& targAdded);


    // Need to override the prob, and it will depend on g if we've made flux cuts
    virtual bool probDependsOnG() const;
    virtual FP prob(const Moment<BC>& mG) const;
    virtual Pqr<BC> probPqr(const typename BC::MDMatrix& dmG) const;
    EIGEN_NEW
  private:
    using Selector<CONFIG>::fluxmin;
    using Selector<CONFIG>::fluxmax;
    using Selector<CONFIG>::jacobian;

    //  Cached intermediate quantities for centroiding case
    FP umin0;  // Y argument offsets
    FP umax0;
    typename BC::MVector vu;  // Y argument coeff for mG
    FP y0;                    // Constant multiple for Y
    typename BC::MVector twidM0;  // Constant vector of J arg
    typename BC::MMatrix CACinv;  // Coeff of mG in J arg
    FP yp0;                       // const multiple of Y'
    typename BC::MVector vyp;     // coeff of mG times Y' 
    FP ypp0;                      // const multiple of Y''
  };
    
  /******************************************************
  Next implementation is for noise that is invariant
  under origin shifts, as in analytic simulations.
  Not relevant for fixed-centroid simulations
  ******************************************************/

  template <class CONFIG>
  class FixedNoiseSelector: public Selector<CONFIG> {
  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    FixedNoiseSelector(const FP fluxmin_,
		       const FP fluxmax_,
		       const TargetGalaxy<BC>& targ);
    virtual FP prob(const Moment<BC>& mG) const;
    virtual bool probDependsOnG() const {return true;}
    virtual Pqr<BC> probPqr(const typename BC::MDMatrix& dmG) const;
    virtual FP detect(const Moment<BC>& mG) const;
    virtual Pqr<BC> detectPqr(const typename BC::MDMatrix& dmG) const;
    EIGEN_NEW
  protected:
    using Selector<CONFIG>::fluxmin;
    using Selector<CONFIG>::fluxmax;
    using Selector<CONFIG>::invsigF;
    using Selector<CONFIG>::jacobian;
  };
    
  /******************************************************
  Next implementation is for noise
  that is invariant under origin shifts, *and* adding noise after selection.
  Not relevant for fixed-centroid cases.
  detect() methods only make sense if called before noise is added,
  and will call the FixedNoise base class.
  ******************************************************/

  template <class CONFIG>
  class AddFixedNoiseSelector: public FixedNoiseSelector<CONFIG> {
  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    AddFixedNoiseSelector(const FP fluxmin_,
			  const FP fluxmax_,
			  const TargetGalaxy<BC>& targ,
			  const TargetGalaxy<BC>& targAdded);
    virtual FP prob(const Moment<BC>& mG) const;
    virtual bool probDependsOnG() const {return true;}
    virtual Pqr<BC> probPqr(const typename BC::MDMatrix& dmG) const;
    EIGEN_NEW
  private:
    using Selector<CONFIG>::fluxmin;
    using Selector<CONFIG>::fluxmax;
    using Selector<CONFIG>::invsigF;
    using Selector<CONFIG>::jacobian;
    // Cached intermediate quantities:
    FP umin0;
    FP umax0;
    typename BC::MVector vu;
  };

} // end namespace bfd

#endif
