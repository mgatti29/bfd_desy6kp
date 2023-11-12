// The moments for BFD, plus covariances of the moments.

#ifndef MOMENT_H
#define MOMENT_H
#include "Std.h"
#include "BfdConfig.h"

namespace bfd {

  // This class represents the measured moments.  The class is templated
  // with a BfdConfig that specifies use case.
  // The moments are split into the centroid (odd parity) and the rest (even)
  template<class CONFIG>
  class Moment {
  public:
    typedef CONFIG BC;
    typedef typename BC::MVector MVector;
    typedef typename BC::XYVector XYVector;
    typedef typename BC::FP FP;
    MVector m;
    XYVector xy;
    Moment(const MVector& _m=MVector(FP(0)),
	   const XYVector& _xy=XYVector(FP(0))): m(_m), xy(_xy) {}
    // Moment<BC>& operator=(const Moment<BC>& rhs) =default;
    //    {m=rhs.m; xy=rhs.xy; return *this;}
    void setZero();
    Moment operator+(const Moment& rhs) const;
    Moment& operator+=(const Moment& rhs);
    Moment operator-(const Moment& rhs) const;
    Moment& operator-=(const Moment& rhs);
    Moment operator*(FP rhs) const;
    Moment& operator*=(FP rhs);
    Moment operator/(FP rhs) const;
    Moment& operator/=(FP rhs);

    // Transform moments for rotation of object ccw by theta:
    void rotate(double theta);
    // Transform moments for flip of object about x axis (y -> -y):
    void yflip();

    EIGEN_NEW
  };

  template<class CONFIG>
  Moment<CONFIG> operator*(typename CONFIG::FP lhs, const Moment<CONFIG>& rhs) {
    return rhs * lhs;
  }

  // Covariance matrix for Moment elements.
  // Keep just the submatrices for XY (odd) elements and other m's
  // since they are uncorrelated.
  template<class CONFIG>
  class MomentCovariance {
  public:
    typedef CONFIG BC;
    typedef typename BC::MMatrix MMatrix;
    typedef typename BC::XYMatrix XYMatrix;
    typedef typename BC::FP FP;

    MMatrix m;
    XYMatrix xy;
    MomentCovariance(const MMatrix& _m=MMatrix(FP(0)),
		     const XYMatrix& _xy=XYMatrix(FP(0))): m(_m), xy(_xy) {}
    MomentCovariance(const MomentCovariance<BC>& rhs) =default;
    MomentCovariance& operator=(const MomentCovariance<BC>& rhs) =default;

    // Arithmetic
    void setZero();
    MomentCovariance operator+(const MomentCovariance& rhs) const;
    MomentCovariance& operator+=(const MomentCovariance& rhs);
    MomentCovariance operator-(const MomentCovariance& rhs) const;
    MomentCovariance& operator-=(const MomentCovariance& rhs);
    MomentCovariance operator*(FP rhs) const;
    MomentCovariance& operator*=(FP rhs);
    MomentCovariance operator/(FP rhs) const;
    MomentCovariance& operator/=(FP rhs);
    
    // Rotate the object (and PSF, everything) CCW by theta, or equivalently
    // rotate the coordinate axes CW by theta
    void rotate(double theta);
    // Transform moments for flip of object about x axis (y -> -y):
    void yflip();
    // Return a new covariance averaged over rotation
    MomentCovariance isotropize() const;
    // Is the matrix isotropic?
    bool isIsotropic(double tolerance=1e-5) const; 
    EIGEN_NEW
  };

  template<class CONFIG>
  MomentCovariance<CONFIG> operator*(typename CONFIG::FP lhs, const MomentCovariance<CONFIG>& rhs) {
    return rhs * lhs;
  }
   
} // end namespace bfd
    
#endif
