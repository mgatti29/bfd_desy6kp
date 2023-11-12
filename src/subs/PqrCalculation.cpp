#include "PqrCalculation.h"

using namespace bfd;

template <class CONFIG>
MLike<CONFIG>::MLike(const TargetGalaxy<BC>& targ,
		     FP sigmaMax):
  mTarget(targ.mom.m),
  invcov(targ.cov.m.inverse()),
  sigmaMaxSquared(sigmaMax*sigmaMax) {}

template <class CONFIG>
typename CONFIG::FP
MLike<CONFIG>::operator()(const typename CONFIG::MVector& mG) const {
  MVector diff = mTarget - mG;
  MVector diffInvcov = invcov * diff;    // A column vector
  FP like= diffInvcov.dot(diff);
  return std::exp(-0.5*like);
}

template <class CONFIG>
Pqr<CONFIG>
MLike<CONFIG>::operator()(const typename BC::MDMatrix& dmG,
			  FP chisqStart) const {
  MVector diff = mTarget - dmG.col(BC::P);
  MVector diffInvcov = invcov * diff;   // MSIZE^2 ops here 
  FP like= diffInvcov.dot(diff);
  if ( like + chisqStart > sigmaMaxSquared) {
    // Signal rejection and return
    Pqr<BC> out;
    out[BC::P] = -1.;
    return out;
  }

  like = exp(-0.5*like);
  // Make a column vector of derivatives:
  Pqr<BC> out(dmG.transpose() * diffInvcov); //MSIZE * DSIZE ops here
  out[BC::P] = 1.;

  // Add Q Q^T to R terms (like what chain rule does)
  out[BC::DG1_DG1] += out[BC::DG1]*out[BC::DG1];
  out[BC::DG2_DG2] += out[BC::DG2]*out[BC::DG2];
  out[BC::DG1_DG2] += out[BC::DG1]*out[BC::DG2];
  if (BC::UseMag) {
    out[BC::DMU_DG1] += out[BC::DMU]*out[BC::DG1];
    out[BC::DMU_DG2] += out[BC::DMU]*out[BC::DG2];
    out[BC::DMU_DMU] += out[BC::DMU]*out[BC::DMU];
  }
  
  // Subtract from 2nd derivs dm * incCov * dm, dm is 1st derivs
  // This part could be cached with Templates if cov is invariant???
  // Takes 3(2) MSIZE^2 + 6(3)*MSIZE mac ops with (without) UseMag.
  MVector vg1(dmG.col(BC::DG1));  // First derivs
  MVector vg2(dmG.col(BC::DG2));  // First derivs
  MVector tmp = invcov * vg1;
  out[BC::DG1_DG1] -= vg1.dot(tmp);
  out[BC::DG1_DG2] -= vg2.dot(tmp);
  tmp = invcov * vg2;
  out[BC::DG2_DG2] -= vg2.dot(tmp);
  if (BC::UseMag) {
    MVector vmu(dmG.col(BC::DMU));  // First derivs
    tmp = invcov * vmu;
    out[BC::DMU_DG1] -= vg1.dot(tmp);
    out[BC::DMU_DG2] -= vg2.dot(tmp);
    out[BC::DMU_DMU] -= vmu.dot(tmp);
  }

  out *= like;
  return out;
}

template <class CONFIG>
XYLike<CONFIG>::XYLike(const TargetGalaxy<BC>& targ) {
  if (BC::FixCenter) return;
  const typename BC::XYMatrix& cxy(targ.cov.xy);
  // Create values such that L(mx, my)
  // = normLxy * exp( mx*mx*invCxx + mx*my*invCxy + my*my*invCyy)
  FP det = cxy(BC::MX,BC::MX)*cxy(BC::MY,BC::MY) -
    cxy(BC::MY,BC::MX)*cxy(BC::MX,BC::MY);
  normLxy = 1. / (2*PI*sqrt(det)); // normalization of xy Gaussian
  invCxx = -0.5*cxy(BC::MY,BC::MY) / det;
  invCyy = -0.5*cxy(BC::MX,BC::MX) / det;
  invCxy = +cxy(BC::MX,BC::MY) / det;
}

template <class CONFIG>
typename CONFIG::FP
XYLike<CONFIG>::operator()(typename BC::XYVector xy) const {
  if (BC::FixCenter) return FP(1.);
  return normLxy * exp(invCxx*xy[BC::MX]*xy[BC::MX] +
		       invCxy*xy[BC::MX]*xy[BC::MY] +
		       invCyy*xy[BC::MY]*xy[BC::MY]);
}

template <class CONFIG>
typename CONFIG::FP
XYLike<CONFIG>::chisq(typename BC::XYVector xy) const {
  if (BC::FixCenter) return FP(0);
  return (invCxx*xy[BC::MX]*xy[BC::MX] +
	  invCxy*xy[BC::MX]*xy[BC::MY] +
	  invCyy*xy[BC::MY]*xy[BC::MY]);
}

template <class CONFIG>
Pqr<CONFIG>
XYLike<CONFIG>::operator()(const typename BC::XYDMatrix& dxy) const {
  if (BC::FixCenter) {
    Pqr<BC> out;
    out[BC::P]=1.;
    return out;
  }
  Pqr<BC> xpqr(dxy.row(BC::MX));
  Pqr<BC> ypqr(dxy.row(BC::MY));

  //Use Pqr multiplication formulae to get derivs
  //of the exponential argument
  Pqr<BC> arg(xpqr);
  Pqr<BC> tmp(invCxx*xpqr + invCxy*ypqr);
  arg *= tmp;
  ypqr *= ypqr;
  arg += invCyy * ypqr;
  // Then use chain rule to do the exponentiation
  FP factor = normLxy * exp(arg[BC::P]);
  return arg.chainRule(factor,factor,factor);
}

template <class CONFIG>
typename CONFIG::FP
XYJacobian<CONFIG>::operator()(const typename BC::MVector& m) const {
  if (BC::FixCenter) return typename BC::FP(1.);
  return 0.25*(m[BC::MR]*m[BC::MR]
	       - m[BC::M1]*m[BC::M1]
	       - m[BC::M2]*m[BC::M2]);
}

template <class CONFIG>
Pqr<CONFIG>
XYJacobian<CONFIG>::operator()(const typename BC::MDMatrix& dm) const {
  if (BC::FixCenter) {
    Pqr<BC> out;
    out[BC::P] = 1.;
    return out;
  }
  Pqr<BC> out(dm.row(BC::MR));
  Pqr<BC> pqr1(dm.row(BC::M1));
  Pqr<BC> pqr2(dm.row(BC::M2));
  out *= out;
  pqr1 *= pqr1;
  pqr2 *= pqr2;
  out -= pqr1;
  out -= pqr2;
  out *= FP(0.25);
  return out;
}

#define INSTANTIATE(...)			   \
  template class bfd::MLike<BfdConfig<__VA_ARGS__> >;	\
  template class bfd::XYLike<BfdConfig<__VA_ARGS__> >;	\
  template class bfd::XYJacobian<BfdConfig<__VA_ARGS__> >;

#include "InstantiateMomentCases.h"
