// Selection probability functions
#include "Selection.h"

using namespace bfd;

// Define this variable to use system erf function instead of rational approximation
// #define USE_ERF

/***************************************************************************
  First a couple of building-block functions:
***************************************************************************/

// Returns the function E(x) and its first N derivatives, defined as
// e(x) = (2*pi)^{-1/2} \int_0^x du exp(-u^2/2) du
//    = 0.5*erf(x/sqrt(2.))
void
eFunc(double x, int N, double &e, double &de, double &d2e, double &d3e, double &d4e) {
  // Use a rational approximation for erf from Abramowitz & Stegun 7.1.25, good to 
  // +-2.5e-5 for all 0\le x < \inf
  // erf(x) = 1 - ( a1*t + a2*t*t + a3*t*t*t) exp(-x^2)
  // t = 1./(1+0.47047 * x)
  // a1 = 0.3480242
  // a2 = -0.0958798
  // a3 = 0.7478556
  const double p = 0.47047 / sqrt(2.);
  const double a1 =  0.3480242 / 2.;
  const double a2 = -0.0958798 / 2.;
  const double a3 =  0.7478556 / 2.;
  const double norm = 1./sqrt(2.*PI);

  bool negate = false;
  if (x < 0) {
    negate = true;
    x = -x;
  }

#ifdef USE_ERF
  e = 0.5*erf(x / sqrt(2.));
#else
  double ee = exp(-0.5*x*x);
  double t = 1./(1. + p * x);
  e = 0.5 - ee*t*(a1 + t*(a2 + t*a3));
#endif

  if (N>0) {
#ifdef USE_ERF
    double ee = exp(-0.5*x*x);
#endif
    de = norm * ee;
    // Do 2nd derivative whenever doing 1st:
    d2e = -x * de;
    if (N>2) {
      // Do 3rd and 4th derivs together
      d3e = (x*x-1) * de;
      d4e = x*(3.-x*x)*de;
    }
  }

  if (negate) {
    e = -e;
    // Just do these multiplications, as quick as checking N:
    d2e = -d2e;
    d4e = -d4e;
  }
}

// This function like eFunc but for the Gaussian integral from xmin to xmax.
// Also allows one to indicate whether there is an min/max bound at all.
void
yFunc(double xmin, double xmax,
      int N,
      double &e, double &de, double &d2e, double &d3e, double &d4e,
      bool hasMin,
      bool hasMax)  {
  // ?? Put SIGMA_CUTOFF in here ??
  double e0=-0.5;
  double de0=0.;
  double d2e0=0.;
  double d3e0=0.;
  double d4e0=0.;
  if (hasMin) eFunc(xmin, N, e0, de0, d2e0, d3e0, d4e0);
  double e1=0.5;
  double de1=0.;
  double d2e1=0.;
  double d3e1=0.;
  double d4e1=0.;
  if (hasMax) eFunc(xmax, N, e1, de1, d2e1, d3e1, d4e1);
  e = e1 - e0;
  if (e==0.) {
    // If e has rounded to zero, make sure others do too.
    de = d2e=d3e=d4e=0.;
    return;
  }
  if (N>0) {
    de = de1 - de0;
    d2e = d2e1 - d2e0;
    if (N>2) {
      d3e = d3e1 - d3e0;
      d4e = d4e1 - d4e0;
    }
  }
  return;
}

  
/***************************************************************************
  Now the base-class functions, which are for case of no added noise.
***************************************************************************/
template<class CONFIG>
Selector<CONFIG>::Selector(const FP fluxmin_,
			   const FP fluxmax_,
			   const TargetGalaxy<BC>& targ): 
  fluxmin(fluxmin_), fluxmax(fluxmax_), jacobian(), A(FP(0)) {
  invsigF = 1./sqrt(targ.cov.m(BC::MF,BC::MF));
  if (BC::FixCenter) return;
    
  jObs = jacobian(targ.mom.m);
  // Make 2 * B * C_Mf vector, where B=diag(0,1/4,-1/4,-1/4)
  A[BC::MF] = 0.;
  A[BC::MR] = 0.5 * targ.cov.m(BC::MR,BC::MF);
  A[BC::M1] = -0.5 * targ.cov.m(BC::M1,BC::MF);
  A[BC::M2] = -0.5 * targ.cov.m(BC::M2,BC::MF);
  // Make scalar b = C_fM * B * C_Mf
  b = 0.25 * ( targ.cov.m(BC::MR,BC::MF)*targ.cov.m(BC::MR,BC::MF)
	       - targ.cov.m(BC::M1,BC::MF)*targ.cov.m(BC::M1,BC::MF)
	       - targ.cov.m(BC::M2,BC::MF)*targ.cov.m(BC::M2,BC::MF) );
}

template<class CONFIG>
bool
Selector<CONFIG>::select(const Moment<BC>& mobs) const {
  // Determine whether an observed galaxy is selected
  // Notice that fluxmin=0 or fluxmax=0 means no selection at that limit.
  if (fluxmin!=0. && mobs.m[BC::MF]<fluxmin) return false;
  if (fluxmax!=0. && mobs.m[BC::MF]>=fluxmax) return false;
  return true;
}
      
template<class CONFIG>
typename CONFIG::FP
Selector<CONFIG>::prob(const Moment<BC>& mG) const {
  // It can be assumed that the observed moments have already passed selection:
  if (BC::FixCenter)
    return 1.;
  else
    return jObs;
}

template<class CONFIG>
bool
Selector<CONFIG>::probDependsOnG() const {
  return false;
}
    
template<class CONFIG>
Pqr<CONFIG>
Selector<CONFIG>::probPqr(const typename BC::MDMatrix& dmG) const {
  Pqr<BC> out;
  out[BC::P]= BC::FixCenter ? 1. : jObs;
  return out;
}

template<class CONFIG>
typename CONFIG::FP
Selector<CONFIG>::detect(const Moment<BC>& mG) const {
  if (BC::FixCenter) {
    if (fluxmin==0. && fluxmax==0.) return 1.;
    double y, dy, d2y, d3y, d4y;
    FP umin = invsigF*(fluxmin - mG.m[BC::MF]);
    FP umax = invsigF*(fluxmax - mG.m[BC::MF]);
    yFunc(umin, umax, 0, y, dy, d2y, d3y, d4y, fluxmin!=0., fluxmax!=0.);
    return y;
  } else {
    FP out = jacobian(mG.m);
    if (fluxmin==0. && fluxmax==0.) return out;
    double z0, z1, z2, z3, z4;
    FP umin = invsigF * (fluxmin - mG.m[BC::MF]);
    FP umax = invsigF * (fluxmax - mG.m[BC::MF]);
    yFunc(umin, umax, 2, z0,z1,z2,z3,z4, fluxmin!=0., fluxmax!=0.);
    z1 *= -invsigF;
    z2 *= invsigF*invsigF;
  
    out *= z0;    // J^G * Y
    out += A.dot(mG.m) * z1;  // - (A*M^G * Y' / sigmaf)
    out += b * z2;  // + b*Y''/sigmaf^2
    return out;
  }
}
      
template<class CONFIG>
Pqr<CONFIG>
Selector<CONFIG>::detectPqr(const typename BC::MDMatrix& dmG) const {
  if (BC::FixCenter) {
    if (fluxmin==0. && fluxmax==0.) {
      // No selection at all, Y=1:
      Pqr<BC> out;
      out[BC::P] = 1.;
      return out;
    }
    FP umin = invsigF*(fluxmin - dmG(BC::MF,BC::P));
    FP umax = invsigF*(fluxmax - dmG(BC::MF,BC::P));
    double y, dy, d2y, d3y, d4y;
    yFunc(umin, umax, 2, y, dy, d2y, d3y, d4y, fluxmin!=0., fluxmax!=0.);

    // Need derivs of Y wrt g
    Pqr<BC> dF(dmG.row(BC::MF));
    dy *= -invsigF;  // since dY/df = dY/du * du/df = -invsigF * dY/du
    d2y*= invsigF*invsigF;
    return dF.chainRule(y,dy,d2y);
  } else {
    // First comes derivatives of the template Jacobian
    Pqr<BC> out = jacobian(dmG);
    // If there is no selection being done, this is all:
    if (fluxmin==0. && fluxmax==0.) {
      return out;
    }

    // More in the Y term if we have active selections
    // Calculate needed derivative of Y
    double z0, z1, z2, z3, z4;
    FP umin = invsigF * (fluxmin - dmG(BC::MF,BC::P));
    FP umax = invsigF * (fluxmax - dmG(BC::MF,BC::P));
    yFunc(umin, umax, 4, z0,z1,z2,z3,z4, fluxmin!=0., fluxmax!=0.);
    
    z1 *= -invsigF;
    z2 *= invsigF*invsigF;
    z3 *= -invsigF*invsigF*invsigF;
    z4 *= invsigF*invsigF*invsigF*invsigF;

    // Now deriv of J^G * Y:
    Pqr<BC> dF(dmG.row(BC::MF));
    out *= dF.chainRule(z0,z1,z2);

    {
      // Add deriv of (A*M^G) * Y'
      Pqr<BC> apqr(dmG.row(BC::MR));
      apqr *= A[BC::MR];
      apqr += A[BC::M1] * Pqr<BC>(dmG.row(BC::M1));
      apqr += A[BC::M2] * Pqr<BC>(dmG.row(BC::M2));
      apqr *= dF.chainRule(z1,z2,z3);
      out += apqr;
    }

    // Add deriv of b * Y''
    out += dF.chainRule(b*z2, b*z3, b*z4);

    return out;
  }
}

/***************************************************************************
  This derived class is for case of adding noise to the moments after selection.
***************************************************************************/

template<class CONFIG>
AddNoiseSelector<CONFIG>::AddNoiseSelector(const FP fluxmin_,
					   const FP fluxmax_,
					   const TargetGalaxy<BC>& targ,
					   const TargetGalaxy<BC>& targAdded):
  Selector<CONFIG>(fluxmin_, fluxmax_, targ)
{
  // In both fixed and varying centroid cases, the argument of the
  // Y function for flux selection is
  // u[min|max] = (f_[min|max] - fG - CM Ctot^-1 (m-mG) )_ff / sigF
  // where sigF^2 = CM_ff - CM_f C^-1 CM_f
  // which we require as
  // u[min|max] = u[min|max]0 + vu . mG
  // with vu = (CM_f Ctot^-1 - U_f) / sigF  (U_f is unit vector at f)
  // u0 = (f_min - CM_f Ctot^-1 m)
  // where m is the moment vector with augmented noise.
  MVector cfm = targ.cov.m.row(BC::MF);
  MMatrix Cinv = targAdded.cov.m.inverse();
  vu = Cinv * cfm;  // Make a column vector
  FP sigf = sqrt(targ.cov.m(BC::MF,BC::MF) - vu.dot(cfm));
  umin0 = (fluxmin - vu.dot(targ.mom.m)) / sigf;
  umax0 = (fluxmax - vu.dot(targ.mom.m)) / sigf;
  vu[BC::MF] -= 1.;
  vu /= sigf;
    
  if (BC::FixCenter) return;

  // In the centroid case, the formula is longer:
  // Y * (J(twidM) + Tr(B CA C^-1 CM) )
  // - Y' * 2 * Z^T B twidM
  // + Y'' Z^T B Z
  // with
  // twidM = CM C^-1 m + CA C^-1 mG
  // Z = ( CA C^-1 CM_f ) / sigf
  //
  // splitting off dependence on mG yields
  // Y * (J(twidM0 + CACinv * mG) + yconst)
  // + Y' * (yp0 + vyp * mG)
  // + Y'' * ypp0
  //
  // twidM0 = CM C^-1 m
  // yconst = Tr(CM B CA C^-1) = -Tr(B CA C^-1 CA)
  // yp0 = -2 Z^T B CM C^-1 M 
  // vyp = -2 Z^T B CA C^-1
  // ypp0 = Z^T B Z 

  // Initialize other quantities to be needed
  MMatrix CA = targAdded.cov.m - targ.cov.m;
  CACinv = CA * Cinv;
  {
    MMatrix tmp = CACinv * targ.cov.m;
    // This is Tr(B CA Cinv CM):
    y0 = 0.25*(tmp(BC::MR,BC::MR)
	       -tmp(BC::M1,BC::M1)
	       -tmp(BC::M2,BC::M2));
  }
  MVector Z = (CACinv * cfm) / sigf; // Has units of M

  // This constant multiplies Y''
  // (Z^T_f B Z_f)
  ypp0 = +0.25*Z[BC::MR]*Z[BC::MR];
  ypp0 += -0.25*Z[BC::M1]*Z[BC::M1];
  ypp0 += -0.25*Z[BC::M2]*Z[BC::M2];

  // Constant term multiplying Y'
  MVector m2ZB(FP(0));  // -2 Z^T B
  m2ZB[BC::MR] = -0.5 * Z[BC::MR];
  m2ZB[BC::M1] += 0.5 * Z[BC::M1];
  m2ZB[BC::M2] += 0.5 * Z[BC::M2];
  yp0 = m2ZB.dot(targ.cov.m * Cinv * targAdded.mom.m);
  // Dot this term into mG then multiply Y'
  vyp = CACinv.transpose() * m2ZB;  // Make vyp =m2ZB*CACinv a column vector

  // This is the constant (independent of mG) part of twidM:
  twidM0 = targ.cov.m * Cinv * targAdded.mom.m;
}  

template<class CONFIG>
bool
AddNoiseSelector<CONFIG>::probDependsOnG() const {
  // prob depends on G if we have any cuts on flux
  // or even without them if centroid is free
  // (also unlikely we would be adding noise without having a flux cut before!)
  return !BC::FixCenter || fluxmin!=0. || fluxmax !=0.;
}
    
template<class CONFIG>
typename CONFIG::FP
AddNoiseSelector<CONFIG>::prob(const Moment<BC>& mG) const {
  if (BC::FixCenter && fluxmin==0. && fluxmax==0.) return 1.; // No selection being made
  FP umin = vu.dot(mG.m);
  FP umax = umax0 + umin;
  umin += umin0;
  double y, dy, d2y, d3y, d4y;

  if (BC::FixCenter) {
    yFunc(umin,umax, 0, y,dy,d2y,d3y,d4y, fluxmin!=0., fluxmax!=0.);
    return y;   // Just the y term for no centering
  }

  // Get the Y term and its derivatives
  yFunc(umin, umax, 2, y,dy,d2y,d3y,d4y, fluxmin!=0., fluxmax!=0.);
  // Make \tilde M = C_A Cinv mG + C_M Cinv mobs:
  MVector twidM = twidM0 + CACinv * mG.m;

  // Here's the master formula:
  return y * (jacobian(twidM) + y0)
    + dy * (yp0 + vyp.dot(mG.m))
    + d2y * ypp0;
}

template<class CONFIG>
Pqr<CONFIG>
AddNoiseSelector<CONFIG>::probPqr(const typename BC::MDMatrix& dmG) const {
  if (BC::FixCenter && fluxmin==0. && fluxmax==0.) {
    // No selection is being made:
    Pqr<BC> out;
    out[BC::P]=1.;
    return out;
  }

  // Get value/derivatives of the argument of Y
  Pqr<BC> vd(dmG.transpose() * vu);
  FP umin = umin0 + vd[BC::P];
  FP umax = umax0 + vd[BC::P];
  double y, dy, d2y, d3y, d4y;
  if (BC::FixCenter) {
    yFunc(umin,umax, 2, y,dy,d2y,d3y,d4y, fluxmin!=0., fluxmax!=0.);
    // propagate derivatives into y function:
    return vd.chainRule(y,dy,d2y);
  }
    
  // Centroid case - need more y derivatives
  yFunc(umin,umax, 4, y,dy,d2y,d3y,d4y, fluxmin!=0., fluxmax!=0.);

  // Make \tilde M = C_A Cinv mG + C_M Cinv mobs and its derivatives
  // (Expensive matrix multiplication here)
  typename BC::MDMatrix dtwidM(CACinv * dmG);
  // Add in the twidM0 terms that matter for Jacobian
  dtwidM(BC::MR,BC::P) += twidM0[BC::MR];
  dtwidM(BC::M1,BC::P) += twidM0[BC::M1];
  dtwidM(BC::M2,BC::P) += twidM0[BC::M2];
    
  // start with derivatives of the template Jacobian
  Pqr<BC> out = jacobian(dtwidM);
  // Now add the Tr term:
  out[BC::P] += y0;
  // Multiply by derivs of Y term
  out *= vd.chainRule(y,dy,d2y);

  // Get deriv of Y' coefficient
  Pqr<BC> dF(dmG.transpose() * vyp);
  // Add in the constant
  dF[BC::P] += yp0;
  // Multiply by derivs of Y'
  dF *= vd.chainRule(dy, d2y, d3y);
  out += dF;

  // Get deriv of Y''
  out += ypp0 * vd.chainRule(d2y, d3y, d4y);

  return out;
}


/****************************************************************
  FixedNoiseSelector is for case when noise is invariant under centroid shift.
  Not applicable to real images!  But useful for analytic tests.
  FixCenter case would be identical to base Selector.
****************************************************************/
template<class CONFIG>
FixedNoiseSelector<CONFIG>::
FixedNoiseSelector(const FP fluxmin_,
		   const FP fluxmax_,
		   const TargetGalaxy<BC>& targ):
  Selector<CONFIG>(fluxmin_, fluxmax_, targ) {
  if (BC::FixCenter)
    throw runtime_error("Should not be creating FixedNoiseSelector for fixed-centroid cases");
}
	       
template<class CONFIG>
typename CONFIG::FP
FixedNoiseSelector<CONFIG>::prob(const Moment<BC>& mG) const {
  return jacobian(mG.m);
}

template <class CONFIG>
Pqr<CONFIG>
FixedNoiseSelector<CONFIG>::probPqr(const typename BC::MDMatrix& dmG) const {
  return jacobian(dmG);
}

template <class CONFIG>
typename CONFIG::FP
FixedNoiseSelector<CONFIG>::detect(const Moment<BC>& mG) const {
  // Start with the detection probability for flux, the Y term
  double y=1.;
  if (fluxmin!=0. || fluxmax!=0.) {
    double dy, d2y, d3y, d4y;
    FP umin = invsigF*(fluxmin - mG.m[BC::MF]);
    FP umax = invsigF*(fluxmax - mG.m[BC::MF]);
    yFunc(umin, umax, 0, y, dy, d2y, d3y, d4y, fluxmin!=0., fluxmax!=0.);
  }
  // times Jacobian
  return jacobian(mG.m) * FP(y);
}
    
template <class CONFIG>
Pqr<CONFIG>
FixedNoiseSelector<CONFIG>::
detectPqr(const typename BC::MDMatrix& dmG) const {
  // Get derivs of Jacobian
  Pqr<BC> out = jacobian(dmG);
  // Multiply by the Y term's derivatives
  if (fluxmin==0. && fluxmax==0.)
    return out;
    
  FP umin = invsigF*(fluxmin - dmG(BC::MF,BC::P));
  FP umax = invsigF*(fluxmax - dmG(BC::MF,BC::P));
  double y, dy, d2y, d3y, d4y;
  yFunc(umin, umax, 2, y, dy, d2y, d3y, d4y, fluxmin!=0., fluxmax!=0.);

  // Need derivs of Y wrt g
  Pqr<BC> dF(dmG.row(BC::MF));
  dy *= -invsigF;  // since dY/df = dY/du * du/df = -invsigF * dY/du
  d2y*= invsigF*invsigF;
  out *= dF.chainRule(y,dy,d2y);
  return out;
}

/************************************************************************
  The artifical case of shift-invariant noise, but now with noise added
  to moments after detection, so prob() function differs.
************************************************************************/
template<class CONFIG>
AddFixedNoiseSelector<CONFIG>::
AddFixedNoiseSelector(const FP fluxmin_,
		      const FP fluxmax_,
		      const TargetGalaxy<BC>& targ,
		      const TargetGalaxy<BC>& targAdded):
  FixedNoiseSelector<BC>(fluxmin_, fluxmax_, targ)
{
  if (BC::FixCenter)
    throw runtime_error("Should not be creating AddFixedNoiseSelector for fixed-centroid cases");

  // The formula for probability in this case is
  // Y * J(mG)
  // where the argument of Y is as in the AddNoise case,
  // u[min|max] = (f_[min|max] - fG - CM Ctot^-1 (m-mG) )_ff / sigF
  // where sigF^2 = CM_ff - CM_f C^-1 CM_f
  // which we require as
  // u[min|max] = u[min|max]0 + vu . mG
  // with vu = (CM_f Ctot^-1 - U_f) / sigF  (U_f is unit vector at f)
  // u0 = (f_min - CM_f Ctot^-1 m)
  // where m is the moment vector with augmented noise.
  typename BC::MVector cfm = targ.cov.m.row(BC::MF);
  typename BC::MMatrix Cinv = targAdded.cov.m.inverse();
  vu = Cinv * cfm; // column vector
  FP sigf = sqrt(targ.cov.m(BC::MF,BC::MF) - vu.dot(cfm));
  umin0 = (fluxmin - vu.dot(targ.mom.m)) / sigf;
  umax0 = (fluxmax - vu.dot(targ.mom.m)) / sigf;
  vu[BC::MF] -= 1.;
    vu /= sigf;
  }
	       
  template<class CONFIG>
  typename CONFIG::FP
  AddFixedNoiseSelector<CONFIG>::prob(const Moment<BC>& mG) const {
    // Start with the Jacobian term
    FP out = jacobian(mG.m);  
    if (fluxmin==0. && fluxmax==0.) return out; // No flux selection being made
    FP umin = vu.dot(mG.m);
    FP umax = umax0 + umin;
    umin += umin0;
    double y, dy, d2y, d3y, d4y;
    yFunc(umin,umax, 0, y,dy,d2y,d3y,d4y, fluxmin!=0., fluxmax!=0.);
    out *= y;
    return out;
  }

  template <class CONFIG>
  Pqr<CONFIG>
  AddFixedNoiseSelector<CONFIG>::
  probPqr(const typename BC::MDMatrix& dmG) const {
    // Start with the Jacobian term
    Pqr<BC> out = jacobian(dmG);
    if (fluxmin==0. && fluxmax==0.) return out; // No flux selection being made
    
    // Then derivatives of Y term
    Pqr<BC> vd(dmG.transpose() * vu);
    FP umin = umin0 + vd[BC::P];
    FP umax = umax0 + vd[BC::P];
    double y, dy, d2y, d3y, d4y;
    yFunc(umin,umax, 2, y,dy,d2y,d3y,d4y, fluxmin!=0., fluxmax!=0.);
    out *= vd.chainRule(y,dy,d2y);
    return out;
  }

#define INSTANTIATE(...) \
  template class bfd::Selector<BfdConfig<__VA_ARGS__>>;	   \
  template class bfd::AddNoiseSelector<BfdConfig<__VA_ARGS__>>; \
  template class bfd::FixedNoiseSelector<BfdConfig<__VA_ARGS__>>; \
  template class bfd::AddFixedNoiseSelector<BfdConfig<__VA_ARGS__>>;
  
#include "InstantiateMomentCases.h"
