#include "Pqr.h"

using namespace bfd;

template<class CONFIG>
Pqr<CONFIG> Pqr<CONFIG>::neglog() const
{
  FP p = getP();
  Pqr<CONFIG> result(*this * FP(-1./p));
  result[BC::P] = -log(p);
  // r = (q^q)/(p*p)-r/p, add the outer product term;
  result[BC::DG1_DG1] += (*this)[BC::DG1]*(*this)[BC::DG1]/(p*p);
  result[BC::DG2_DG2] += (*this)[BC::DG2]*(*this)[BC::DG2]/(p*p);
  result[BC::DG1_DG2] += (*this)[BC::DG1]*(*this)[BC::DG2]/(p*p);
  if (BC::UseMag) {
    result[BC::DMU_DG1] += (*this)[BC::DMU]*(*this)[BC::DG1]/(p*p);
    result[BC::DMU_DG2] += (*this)[BC::DMU]*(*this)[BC::DG2]/(p*p);
    result[BC::DMU_DMU] += (*this)[BC::DMU]*(*this)[BC::DMU]/(p*p);
  }
  return result;
}

template<class CONFIG>
const Pqr<CONFIG>& 
Pqr<CONFIG>::operator*=(const Pqr<CONFIG>& rhs) {
  if (&rhs==this) {
    // Squaring self
    FP TWO = 2.;
    FP p2 = TWO*getP();
    this->subVector(BC::R,BC::DSIZE) *= p2;
    (*this)[BC::DG1_DG1] += TWO * (*this)[BC::DG1] * (*this)[BC::DG1];
    (*this)[BC::DG1_DG2] += TWO * (*this)[BC::DG1] * (*this)[BC::DG2];
    (*this)[BC::DG2_DG2] += TWO * (*this)[BC::DG2] * (*this)[BC::DG2];
    if (BC::UseMag) {
      (*this)[BC::DMU_DG1] += TWO * (*this)[BC::DMU] * (*this)[BC::DG1];
      (*this)[BC::DMU_DG2] += TWO * (*this)[BC::DMU] * (*this)[BC::DG2];
      (*this)[BC::DMU_DMU] += TWO * (*this)[BC::DMU] * (*this)[BC::DMU];
    }
    this->subVector(BC::Q,BC::R) *= p2;
    (*this)[BC::P] *= (*this)[BC::P];
    return *this;
  }
  FP p = getP();
  FP q = rhs.getP();
  // r = lhs.p * rhs.r + lhs.r * rhs.p + lhs.q ^ rhs.q + rhs.q ^ lhs.q
  // where ^ is outer product.
  this->subVector(BC::R,BC::DSIZE) *= q;
  this->subVector(BC::R,BC::DSIZE) += p*rhs.subVector(BC::R,BC::DSIZE);
  (*this)[BC::DG1_DG2] += rhs[BC::DG1]*(*this)[BC::DG2] + rhs[BC::DG2]*(*this)[BC::DG1];
  (*this)[BC::DG1_DG1] += FP(2.)*rhs[BC::DG1]*(*this)[BC::DG1];
  (*this)[BC::DG2_DG2] += FP(2.)*rhs[BC::DG2]*(*this)[BC::DG2];
  if (BC::UseMag) {
    (*this)[BC::DMU_DG1] += rhs[BC::DMU]*(*this)[BC::DG1] + rhs[BC::DG1]*(*this)[BC::DMU];
    (*this)[BC::DMU_DG2] += rhs[BC::DMU]*(*this)[BC::DG2] + rhs[BC::DG2]*(*this)[BC::DMU];
    (*this)[BC::DMU_DMU] += FP(2.)*rhs[BC::DMU]*(*this)[BC::DMU];
  }
  // q = lhs.p * rhs.q + lhs.q * rhs.p
  this->subVector(BC::Q,BC::R) *= q;
  this->subVector(BC::Q,BC::R) += p*rhs.subVector(BC::Q,BC::R);
  // p = lhs.p * rhs.p
  (*this)[BC::P] = p*q;
  return *this;
}

template<class CONFIG>
void Pqr<CONFIG>::getG(typename BC::QVector &g,
		       typename BC::RMatrix &cov) const
{
  // Catch tmv error (degenerate matrix) and re-throw with specifics
  try {
    cov=getR().inverse();
    g=-(cov*getQ());
  } catch (tmv::Error& e) {    
    throw std::runtime_error("Matrix inversion error for Pqr::getG");
  }
}

template<class CONFIG>
Pqr<CONFIG>
Pqr<CONFIG>::chainRule(FP f, FP df, FP ddf) const {
    Pqr out;
    out[BC::P] = f;
    out.subVector(BC::Q,BC::DSIZE) = df * this->subVector(BC::Q,BC::DSIZE);
    out[BC::DG1_DG1] += ddf*(*this)[BC::DG1]*(*this)[BC::DG1];
    out[BC::DG1_DG2] += ddf*(*this)[BC::DG1]*(*this)[BC::DG2];
    out[BC::DG2_DG2] += ddf*(*this)[BC::DG2]*(*this)[BC::DG2];
    if (BC::UseMag) {
      out[BC::DMU_DG1] += ddf*(*this)[BC::DMU]*(*this)[BC::DG1];
      out[BC::DMU_DG2] += ddf*(*this)[BC::DMU]*(*this)[BC::DG2];
      out[BC::DMU_DMU] += ddf*(*this)[BC::DMU]*(*this)[BC::DMU];
    }
    return out;
  }
  
template <class CONFIG>
void Pqr<CONFIG>::rotate(double theta)
{
  DComplex z2(cos(2.*theta),sin(2.*theta));
  
  DComplex ztmp = DComplex( (*this)[BC::DG1],(*this)[BC::DG2] ) * z2;
  (*this)[BC::DG1] = real(ztmp);  
  (*this)[BC::DG2] = imag(ztmp);
  if (BC::UseMag) {
    ztmp = DComplex( (*this)[BC::DMU_DG1],(*this)[BC::DMU_DG2] ) * z2;
    (*this)[BC::DMU_DG1] = real(ztmp);  
    (*this)[BC::DMU_DG2] = imag(ztmp);
  }
  double r0 = 0.5* ( (*this)[BC::DG1_DG1] + (*this)[BC::DG2_DG2] );
  DComplex r2(0.5* ( (*this)[BC::DG1_DG1] - (*this)[BC::DG2_DG2] ), (*this)[BC::DG1_DG2]);
  r2 *= z2*z2;
  (*this)[BC::DG1_DG1] = r0 + real(r2);
  (*this)[BC::DG2_DG2] = r0 - real(r2);
  (*this)[BC::DG1_DG2] = imag(r2);
}

// Instantiate the class

#define INSTANTIATE(...)     \
  template class Pqr<BfdConfig<__VA_ARGS__> >;
#include "InstantiateMomentCases.h"
