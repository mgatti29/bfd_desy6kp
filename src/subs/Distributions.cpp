#include "Distributions.h"
#ifdef USE_TMV
#include "tmv/TMV_SymCHD.h"
#elif defined USE_EIGEN
#include "Eigen/Cholesky"
#endif

using namespace bfd;

// Constructor gets the Cholesky decomposition, do it differently in TMV and Eigen.
#ifdef USE_TMV
template<class T>
MultiGauss<T>::MultiGauss(const  Matrix& cov_): dim(cov_.nrows())
{
  tmv::SymMatrix<T> covCopy(cov_);
  A.resize(dim);
  tmv::CH_Decompose(covCopy);
  A=covCopy.lowerTri();
}
#elif defined USE_EIGEN
template<class T>
MultiGauss<T>::MultiGauss(const  Matrix& cov_):
  dim(cov_.nrows()),
  A(Eigen::LLT<typename Matrix::Base,Eigen::Lower>(cov_.template selfadjointView<Eigen::Lower>()).matrixL())
{}
#endif

template<class T>
linalg::Vector<T> MultiGauss<T>::sample(ran::GaussianDeviate<double> &rand) const
{
  Vector vec(dim);
  for(int i=0;i<dim;++i) vec(i)=rand();
#ifdef USE_TMV
  return Vector(A*vec);
#elif defined USE_EIGEN
  // Slight acceleration from specifying lower triangle only present
  return Vector(A.template triangularView<Eigen::Lower>()*vec);
#endif
}

template<class T>
linalg::Vector<T> MultiGauss<T>::sample(const Vector& mean,
			     ran::GaussianDeviate<double> &rand) const
{
  return Vector(sample(rand) + mean);
}

void
BobsEDist::sample(double &e1, double& e2) {
  double esq;
  do {
    do {
      // First draw from 2d Gaussian truncated at |r|=1
      e1 = gd()*sigma;
      e2 = gd()*sigma;
      esq = e1*e1 + e2*e2;
    } while (esq>=1);
    // Then use rejection to shape distribution by (1-r^2)^2 factor:
  } while (ud() > (1-esq)*(1-esq));
  return;
}

PowerLawDistribution::PowerLawDistribution(double min, double max, 
					   double gamma,
					   ran::UniformDeviate<double>& ud_): ud(ud_),
									      gammaPlus1(gamma+1.) {
  // Cumulative distribution is P(<x) = x^(gamma+1) - min^(gamma+1) /
  //                                    max^(gamma+1) - min^(gamma+1) 
  // Or, if gamma=-1, ln(x/min) / ln(max/min).
  if (gammaPlus1==0.) {
    base = log(min);
    norm = log(max/min);
  } else {
    base = pow(min, gammaPlus1);
    norm = pow(max, gammaPlus1) - base;
  }
}

double
PowerLawDistribution::sample() {
  double v = ud() * norm + base;
  return gammaPlus1==0. ? exp(v) : pow(v, 1./gammaPlus1);
}

template class bfd::MultiGauss<double>;
template class bfd::MultiGauss<float>;		
