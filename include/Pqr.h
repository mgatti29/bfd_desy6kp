#ifndef PQR_H
#define PQR_H

#include "Std.h"
#include "LinearAlgebra.h"
#include "BfdConfig.h"

namespace bfd {

  template<class CONFIG>
  class Pqr: public CONFIG::PqrVector {
  public:
    typedef CONFIG BC;
    typedef typename CONFIG::FP FP;
    Pqr(): BC::PqrVector(FP(0)) {}
    // Construct from either the base class vector (which should be
    // a fixed-size linalg::SVector of matched type) or a generic
    // double-valued dynamic-sized linalg::Vector
    Pqr(const typename BC::PqrVector& v): BC::PqrVector(v) {}
    // ambiguous conversions from this?? Pqr(const linalg::DVector& v): BC::PqrVector(v) {}
    const Pqr& operator=(const typename BC::PqrVector& v) {
      if (this != &v) BC::PqrVector::operator=(v);
      return *this;
    }
    const Pqr& operator=(const linalg::DVector& v) {
      BC::PqrVector::operator=(v);
      return *this;
    }

    FP getP() const {return (*this)[BC::P];}
    typename BC::QVector getQ() const {return this->subVector(BC::Q,BC::R);}
    typename BC::RMatrix getR() const {
      typename BC::RMatrix r;
      const int iG1 = BC::DG1-BC::Q;
      const int iG2 = BC::DG2-BC::Q;
      r(iG1,iG1) = (*this)[BC::DG1_DG1];
      r(iG2,iG2) = (*this)[BC::DG2_DG2]; 
      r(iG1,iG2) = (*this)[BC::DG1_DG2];
      r(iG2,iG1) = (*this)[BC::DG1_DG2];
      if (BC::UseMag) {
	const int iMu = BC::DMU-BC::Q;
	r(iMu,iG1) = (*this)[BC::DMU_DG1];
	r(iG1,iMu) = (*this)[BC::DMU_DG1];
	r(iMu,iG2) = (*this)[BC::DMU_DG2];
	r(iG2,iMu) = (*this)[BC::DMU_DG2];
	r(iMu,iMu) = (*this)[BC::DMU_DMU];
      }
      return r;
    }

    void setP(FP p)  {(*this)[BC::P] =p;}
    void setQ(typename BC::QVector& q) {this->subVector(BC::Q,BC::R) = q;}
    void setR(typename BC::RMatrix& r) {
      const int iG1 = BC::DG1-BC::Q;
      const int iG2 = BC::DG2-BC::Q;
      (*this)[BC::DG1_DG1]=r(iG1,iG1);
      (*this)[BC::DG2_DG2]=r(iG2,iG2); 
      (*this)[BC::DG1_DG2]=r(iG1,iG2);
      if (BC::UseMag) {
	const int iMu = BC::DMU-BC::Q;
	(*this)[BC::DMU_DG1]=r(iMu,iG1);
	(*this)[BC::DMU_DG2]=r(iMu,iG2); 
	(*this)[BC::DMU_DMU]=r(iMu,iMu);
      }
    }

#ifndef USE_TMV
    // Need to import scalar *= from base class because else it will be hidden
    // by the *=(Pqr) operator defined below.
    // TMV implements *= as an external binop, not as class member, so this
    // is neither necessary nor allowed.
    using BC::PqrVector::operator*=;
#endif
    
    // Convert this into the Taylor expansion of P * rhs.P:
    inline
    const Pqr& operator*=(const Pqr& rhs) {
      FP TWO = 2.;
      if (&rhs==this) {
	// Squaring self
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
      (*this)[BC::DG1_DG1] += TWO*rhs[BC::DG1]*(*this)[BC::DG1];
      (*this)[BC::DG2_DG2] += TWO*rhs[BC::DG2]*(*this)[BC::DG2];
      if (BC::UseMag) {
	(*this)[BC::DMU_DG1] += rhs[BC::DMU]*(*this)[BC::DG1] + rhs[BC::DG1]*(*this)[BC::DMU];
	(*this)[BC::DMU_DG2] += rhs[BC::DMU]*(*this)[BC::DG2] + rhs[BC::DG2]*(*this)[BC::DMU];
	(*this)[BC::DMU_DMU] += TWO*rhs[BC::DMU]*(*this)[BC::DMU];
      }
      // q = lhs.p * rhs.q + lhs.q * rhs.p
      this->subVector(BC::Q,BC::R) *= q;
      this->subVector(BC::Q,BC::R) += p*rhs.subVector(BC::Q,BC::R);
      // p = lhs.p * rhs.p
      (*this)[BC::P] = p*q;
      return *this;
    }
    
    // Return Taylor expansion of f(p) if this is the Taylor
    // expansion of p and we are given the values of f(p),
    // f'(p), and f''(p).
    inline
    Pqr chainRule(FP f, FP df, FP ddf) const {
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
    
    // Return PQR expansion of -log of this PQR expansion
    inline
    Pqr neglog() const {
      FP p = getP();
      Pqr<CONFIG> result(*this * FP(-1./p));
      result[BC::P] = -std::log(p);
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
      
    // Return estimate of g and its covariance, assuming this is Pqr of -ln(probability)
    void getG(typename BC::QVector &g, typename BC::RMatrix &cov) const {
      try {
	cov=getR().inverse();
	g=-(cov*getQ());
      } catch (std::exception& e) {    
	throw std::runtime_error("Matrix inversion error for Pqr::getG");
      }
    }

    // Rotate the object by +theta, or equivalently rotate the axes defining
    // G1 and G2 CW by -theta.
    void rotate(double theta) {
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

    // Evaluate the Taylor expansion at the lensing values given by the vector g
    FP operator()(const typename BC::QVector &g) const {
      FP result = (*this)[BC::P]
	+ (*this)[BC::DG1] * g[BC::G1]
	+ (*this)[BC::DG2] * g[BC::G2]
	+ (*this)[BC::DG1_DG1] * 0.5 * g[BC::G1] * g[BC::G1]
	+ (*this)[BC::DG2_DG2] * 0.5 * g[BC::G2] * g[BC::G2]
	+ (*this)[BC::DG1_DG2] * g[BC::G1] * g[BC::G2];
      if (BC::UseMag)
	result += g[BC::MU] * ( (*this)[BC::DMU] 
				+ (*this)[BC::DMU_DG1] * g[BC::G1]
				+ (*this)[BC::DMU_DG2] * g[BC::G2]
				+ (*this)[BC::DMU_DMU] * 0.5 * g[BC::MU]);
      return result;
    }
    FP operator()(const linalg::DVector &g) const {
      return this->operator()(typename BC::QVector(g));
    }
  };

} // end namespace bfd


#endif
