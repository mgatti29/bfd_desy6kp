// Declarations of classes describing random distributions

#ifndef DISTRIBUTIONS_H
#define DISTRIBUTIONS_H

#include "LinearAlgebra.h"
#include "Std.h"
#include <memory>
#include "Random.h"

namespace bfd {

  // Note that the sampling routines here are NOT thread-safe unless the
  // random-number generators are set up to be so.

  // Multivariate Normal class to do sampling
  template<class T>
  class MultiGauss{
  public:
    typedef typename linalg::Matrix<T> Matrix;
    typedef typename linalg::Vector<T> Vector;
    MultiGauss(const Matrix& cov_);
    // The lower triangular part of cov_ will be assumed to define the symmetric cov matrix
    Vector sample(const Vector &_mean,
		  ran::GaussianDeviate<double> &rand) const;

    // Zero-mean case:
    Vector  sample(ran::GaussianDeviate<double> &rand) const; 
    // Get the covariance (in plain-old-matrix form)
    Matrix cov() const {return Matrix(A * A.transpose());}
    EIGEN_NEW
  private:
    int dim;
    // store cholesky decomposition matrix
#ifdef USE_TMV
    tmv::LowerTriMatrix<T> A;
#elif defined USE_EIGEN
    // Store the lower triangular part in a full matrix
    typename Matrix::Base A;
#endif
  };

  // Distribution function over e1, e2 which is
  // P \propto \exp(-e^2/2*\sigma^2) (1-e^2)^2
  // for 0 <=e < 1.
  class BobsEDist {
  public:
    BobsEDist(double sigma_,
	      ran::UniformDeviate<double> &ud_): sigma(sigma_), ud(ud_), gd(ud_) {}
    void sample(double& e1, double& e2);
  private:
    double sigma;
    ran::UniformDeviate<double> &ud;
    ran::GaussianDeviate<double> gd;
  };

  class PowerLawDistribution {
    // Class which will sample values between min and max with probability dP/dx \propto x^gamma
  public:
    PowerLawDistribution(double min, double max, double gamma,
			 ran::UniformDeviate<double>& ud_);
    double sample();
  private:
    ran::UniformDeviate<double>& ud;
    double gammaPlus1;  // 1./(1+gamma)
    double norm;
    double base;
  };

} // end namespace bfd

#endif // DISTRIBUTIONS_H
