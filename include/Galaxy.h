#ifndef GALAXY_H
#define GALAXY_H

#include "LinearAlgebra.h"
#include "BfdConfig.h"
#include "Moment.h"
#include "Random.h"
#include "Distributions.h"

namespace bfd
{

  // Target galaxy structure.
  template<class CONFIG>
  class TargetGalaxy {
  public:
    typedef CONFIG BC;
    TargetGalaxy(const Moment<CONFIG>& mom_ = Moment<CONFIG>(),
		 const MomentCovariance<CONFIG>& cov_ = MomentCovariance<CONFIG>(),
		 const linalg::DVector2& position_ = linalg::DVector2(0.),
		 const long id_=0): mom(mom_), cov(cov_),
				  position(position_),
				  id(id_) {}
    Moment<CONFIG> mom;
    MomentCovariance<CONFIG> cov;
    long id;
    linalg::DVector2 position;  // xy coordinates of galaxy
    // Rotate galaxy by theta (or coordinates by -theta). Position is unchanged.
    void rotate(double theta) {mom.rotate(theta); cov.rotate(theta);}
    // Flip galaxy on x axis, y -> -y
    void yflip() {mom.yflip(); cov.yflip();}
    EIGEN_NEW
  };

  // Template galaxy structure. It will store moments and derivatives
  // in their complex-valued forms to enable easy transformations.
  // So the class has its own set of indices for moments and derivs
  template <class CONFIG>
  class TemplateGalaxy {
  public:
    typedef CONFIG BC;
    typedef typename CONFIG::FP FP;
    typedef std::complex<FP> FPC;

    // Indices into complex array of moments
    static const int MF = 0;                      // Flux moment
    static const int MR = MF+1;                   // Radius/size moment |k^2|
    static const int ME = MR+1;                   // Ellipticity moment M1 + iM2
    static const int MC = BC::UseConc ? ME+1 : -1;    // Concentration moment |k^4|
    static const int MC0 = BC::UseConc ? MC+1 : ME+1; // Index of first color
    static const int MSIZE = MC0 + BC::Colors;      // Total size of (non-centroid) moment vector

    static const int MX = BC::FixCenter ? -1 : 0;     // Index of X + iY moment
    static const int XYSIZE = BC::FixCenter ? 1 : 1;  // Size of XY moment vector - must be >0 for tmv::SmallMatrix

    // Indices for Taylor expansions w.r.t. lensing
    static const int D0 = 0;                      // Index of 0th derivative, i.e. value
    static const int Q = 1;                       // Index where 1st derivs start
    static const int DU = BC::UseMag ? Q : -1;    // Deriv wrt magnification
    static const int DV = BC::UseMag? DU+1 : Q;   // DG1 + i DG2
    static const int DVb = DV+1;                  // DG1 - i DG2
    static const int R = DVb + 1;                 // First index of 2nd derivs
    static const int DU_DU = BC::UseMag? R : -1;
    static const int DU_DV = BC::UseMag? DU_DU+1 : -1;
    static const int DU_DVb = BC::UseMag? DU_DV+1 : -1;
    static const int DV_DV = BC::UseMag? DU_DVb+1 : R;
    static const int DVb_DVb = DV_DV+1;
    static const int DV_DVb = DVb_DVb+1;
    static const int DSIZE =  DV_DVb+1; // Total dimension of moments + derivs

    typedef linalg::Matrix<std::complex<FP>> DerivMatrix;
    DerivMatrix mDeriv; // Even moments and derivatives
    DerivMatrix xyDeriv; // Odd (centroid) moments and derivatives
    FP nda;  // Weight = (sky density) * (dA of centroid shifting, if done)
    long id; // ID of original source galaxy
    FP jSuppression;  // ratio of Jacobian here to Jacobian at posn with M_XY=0

    EIGEN_NEW
      
    TemplateGalaxy(const DerivMatrix& mderiv_ = DerivMatrix(MSIZE,DSIZE,0.),
		   const DerivMatrix& xyderiv_ = DerivMatrix(XYSIZE,DSIZE,0.),
		   const FP nda_ = 1.,
		   const long id_ = 0L,
		   const FP jSuppression_=1.):
    mDeriv(mderiv_),
    xyDeriv(xyderiv_),
    nda(nda_),
    id(id_),
    jSuppression(jSuppression_)
    {
      Assert(mDeriv.nrows()==MSIZE && mDeriv.ncols()==DSIZE);
      Assert(xyDeriv.nrows()==XYSIZE && xyDeriv.ncols()==DSIZE);
    }

    // Rotate galaxy by theta (or coordinates by -theta)
    void rotate(double theta);
    // Flip galaxy on x axis, y -> -y
    void yflip();

    // Return moments and derivatives in real-valued forms
    typename BC::MDMatrix realMDerivs() const;
    typename BC::XYDMatrix realXYDerivs() const;
  };

  // GalaxyData is an interface to anything that can produce
  // TargetGalaxy or TemplateGalaxy information.
  template <class CONFIG>
  class GalaxyData {
  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    virtual ~GalaxyData() {}
    // Create a TargetGalaxy from the data, with or w/o covariance
    virtual TargetGalaxy<CONFIG> getTarget(bool fillCovariance=true) const=0;
    // Create a TemplateGalaxy from the data.
    virtual TemplateGalaxy<CONFIG> getTemplate() const=0;
    // Return new data with coordinate origin moved by (dx,dy)
    virtual GalaxyData* getShifted(double dx, double dy) const=0;
    // Return new data with origin shifted to null the XY moments
    // Base class implements Newton iteration.
    // maxShift is the farthest that X & Y will be allowed to go.
    // Returns null if it wanders beyond there or fails to meet
    // convergence criteria explained in code.
    virtual GalaxyData* getNullXY(FP maxShift=1.) const;
    // Return Jacobian derivative matrix of XY moments vs X,Y shifts.
    // Default implementation uses 2nd moments
    virtual typename BC::XYMatrix xyJacobian() const;
    // Return a vector of TemplateGalaxies sampling grid of XY shifts.
    // xySigma, fluxSigma = assumed error on XY and flux moments
    // fluxMin = minimum flux to be used in integrations (0 for no limit)
    // sigmaStep, sigmaMax = interval and maximum sigma used in integrations
    // xyMax = farthest to translate origin from null point
    vector<TemplateGalaxy<BC>> getTemplateGrid(ran::UniformDeviate<double>& ud,
					       FP xySigma,
					       FP fluxSigma,
					       FP fluxMin=0.,
					       FP sigmaStep=1.,
					       FP sigmaMax=6.5,
					       FP xyMax=2.) const;
    // ??? Need specialized vector class to handle Eigen alignment ???
  };

  // Class for a GalaxyData plus a fixed moment noise vector
  // generated from its (or a supplied) covariance matrix.
  // Noise is *not* added for getTemplate() call, i.e. derivs are noiseless.
  // The GalaxyData it wraps must stay in existence!
  template <class CONFIG>
  class GalaxyPlusNoise: public GalaxyData<CONFIG> {
  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    GalaxyPlusNoise(const GalaxyData<BC>* gptr_,
		    ran::GaussianDeviate<double>& gd,
		    const MultiGauss<FP>* noiseGen=0);
    virtual ~GalaxyPlusNoise() {}
    
    virtual TargetGalaxy<CONFIG> getTarget(bool fillCovariance=true) const {
      auto out = gptr->getTarget(false);  // Assuming noiseless parent, no cov
      out.mom += noise;
      out.cov = cov;
      return out;
    }
    virtual TemplateGalaxy<CONFIG> getTemplate() const {
      return gptr->getTemplate();
    }
    virtual GalaxyData<BC>* getShifted(double dx, double dy) const {
      auto out = new GalaxyPlusNoise<BC>(*this);
      out->gptr = gptr->getShifted(dx,dy);
      return out;
    }      
    virtual typename BC::XYMatrix xyJacobian() const {
      return gptr->xyJacobian(); // Added noise is invariant under shift
    }
    EIGEN_NEW
  private:
    const GalaxyData<BC>* gptr;
    Moment<BC> noise;
    MomentCovariance<BC> cov;
  };
} // end namespace bfd

#endif // GALAXY_H
