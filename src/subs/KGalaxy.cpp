// Code to get the moments and derivatives from sampled k-space data

#include "KGalaxy.h"
#include <list>

using namespace bfd;

// Construct with no variance given
template <class CONFIG>
KGalaxy<CONFIG>::KGalaxy(const KWeight<FP>& kw,
			 const CVector& kval_,
			 const Vector& kx,
			 const Vector& ky,
			 FP d2k,
			 const std::set<int>& unconjugated,
			 const linalg::DVector2& posn_,
			 const long id_): wt(&kw),
					  kvar(nullptr),
					  posn(posn_),
					  id(id_)
{
  const std::complex<FP> I(0,1);
  Assert(ky.size()==kx.size());
  Assert(kval_.size()==kx.size());
  Assert(kvar_.size()==kx.size());
  Vector wts = kw(kx*kx + ky*ky);
  // Resize arrays to use only the nonzero weight points:
  int nkeep = 0;
  for (int i=0; i<kx.size(); i++)
    if (wts[i]!=0) nkeep++;
  kval.resize(nkeep); // ??? Need to override resize with single argument in LinearAlgebra ??
  kz.resize(nkeep);
  int iout = 0;
  for (int i=0; i<kx.size(); i++) {
    if (wts[i]==0.) continue;  // Don't use zero-weighted data
    kval[iout] = kval_[i] * d2k * FP(2);
    if (unconjugated.count(i)) kval[iout] *= FP(0.5);  // only count these once in integral
    kz[iout] = kx[i] + I*ky[i];
    iout++;
  }
}

// Construct with variance given
template <class CONFIG>
KGalaxy<CONFIG>::KGalaxy(const KWeight<FP>& kw,
			 const CVector& kval_,
			 const Vector& kx,
			 const Vector& ky,
			 const Vector& kvar_,
			 FP d2k,
			 const std::set<int>& unconjugated,
			 const linalg::DVector2& posn_,
			 const long id_): wt(&kw),
					  kvar(nullptr),
					  posn(posn_),
					  id(id_)
{
  const std::complex<FP> I(0,1);
  Assert(ky.size()==kx.size());
  Assert(kval_.size()==kx.size());
  Assert(kvar_.size()==kx.size());
  Vector wts = kw(kx*kx + ky*ky);
  // Resize arrays to use only the nonzero weight points:
  int nkeep = 0;
  for (int i=0; i<kx.size(); i++)
    if (wts[i]!=0) nkeep++;
  kval.resize(nkeep); // ??? Need to override resize with single argument in LinearAlgebra ??
  kz.resize(nkeep);
  kvar = new Vector(nkeep);
  int iout = 0;
  for (int i=0; i<kx.size(); i++) {
    if (wts[i]==0.) continue;  // Don't use zero-weighted data
    kval[iout] = kval_[i] * d2k * FP(2);
    (*kvar)[iout] = kvar_[i] * d2k * d2k * FP(2);
    if (unconjugated.count(i)) {
      kval[iout] *= FP(0.5);  // only count these once in integral
      (*kvar)[iout] *= FP(0.5);  // only count these once in integral
    }
    kz[iout] = kx[i] + I*ky[i];
    iout++;
  }
}

// Copy
template <class CONFIG>
KGalaxy<CONFIG>::KGalaxy(const KGalaxy<CONFIG>& rhs): kval(rhs.kval),
						      kz(rhs.kz),
						      kvar(nullptr),
						      wt(rhs.wt),
						      posn(rhs.posn),
						      id(rhs.id) {
  if (rhs.kvar) kvar = new Vector(*rhs.kvar);
}

// move
template <class CONFIG>
KGalaxy<CONFIG>::KGalaxy(KGalaxy<CONFIG>&& rhs): kval(std::move(rhs.kval)),
						 kz(std::move(rhs.kz)),
						 kvar(rhs.kvar),
						 wt(rhs.wt),
						 posn(rhs.posn),
						 id(rhs.id) {
  rhs.kvar = nullptr;
}

// assignment
template <class CONFIG>
KGalaxy<CONFIG>&
KGalaxy<CONFIG>::operator=(const KGalaxy<CONFIG>& rhs) {
  kval.resize(rhs.kval.size());
  kval = rhs.kval;
  kz.resize(rhs.kval.size());
  kz = rhs.kz;
  delete kvar; kvar=nullptr;
  if (rhs.kvar) kvar = new Vector(*rhs.kvar);
  wt = rhs.wt;
  posn = rhs.posn;
  id = rhs.id;
  return *this;
}

// move assignment
template <class CONFIG>
KGalaxy<CONFIG>&
KGalaxy<CONFIG>::operator=(KGalaxy<CONFIG>&& rhs) {
  kval = std::move(rhs.kval);
  kz = std::move(rhs.kz);
  delete kvar; kvar=nullptr;
  if (rhs.kvar) {kvar = new Vector(*rhs.kvar); rhs.kvar=nullptr;}
  wt = rhs.wt;
  posn = rhs.posn;
  id = rhs.id;
  return *this;
}

template <class CONFIG>
KGalaxy<CONFIG>*
KGalaxy<CONFIG>::getShifted(double dx, double dy) const {
  // Note that dx,dy are shift of coordinate origin so phase is e^ikx.
  const std::complex<FP> I(0,1);
  auto out = new KGalaxy<CONFIG>(*this);
  Vector phi = FP(dx)*kz.REAL + FP(dy)*kz.IMAG;
  // out = in * exp(I*phi):
#ifdef USE_TMV
  for (int i=0; i<kval.size(); i++) {
    std::complex<FP> z(sin(phi[i]),cos(phi[i]));
    out->kval[i] *= z;
  }
#elif defined USE_EIGEN
  // Potentially vectorized calls
  CVector z(phi.size());
  z.imag() = phi.array().sin();
  z.real() = phi.array().cos();
  out->kval = kval.cwiseProduct(z);
#endif
  return out;
}

template <class CONFIG>
TargetGalaxy<CONFIG>
KGalaxy<CONFIG>::getTarget(bool fillCovariance) const {

  // First build the matrix that holds coefficients of even moments
  Matrix wtF(BC::MSIZE, kval.size(), 0.);
  Vector ksq = kz.cwiseProduct(kz.adjoint()).REAL;  // |k^2|
  Vector vw = (*wt)(ksq);
  wtF.row(BC::MF) = vw;
  wtF.row(BC::MR) = vw.cwiseProduct(ksq);
  if (BC::UseConc) {
    wtF.row(BC::MC) = ksq.cwiseProduct(wtF.row(BC::MR));
  }
  CVector kk = kz.cwiseProduct(kz);
  wtF.row(BC::M1) = vw.cwiseProduct(kk.REAL);
  wtF.row(BC::M2) = vw.cwiseProduct(kk.IMAG);
  
  // Now apply to data to get moments
  Moment<BC> mom;
  mom.m = wtF * kval.REAL;
  MomentCovariance<BC> cov;
  if (fillCovariance) {
    // And get covariance if requested too
    if (!kvar)
      throw std::runtime_error("KGalaxy::getTarget call with fillCovariance but"
			       " no kvar given");
#ifdef USE_TMV
    cov.m = wtF * tmv::DiagMatrix<FP>(*kvar) * wtF.transpose();
#elif defined USE_EIGEN
    cov.m = wtF * kvar->asDiagonal() * wtF.transpose();
#endif
  }
  
  // Now the odd moments, if in use
  if (!BC::FixCenter) {
    wtF.resize(BC::XYSIZE, kval.size());
    wtF.row(BC::MX) = vw.cwiseProduct(kz.REAL);
    wtF.row(BC::MY) = vw.cwiseProduct(kz.IMAG);
    mom.xy = -wtF * kval.IMAG;
  if (fillCovariance) 
#ifdef USE_TMV
    cov.xy = wtF * tmv::DiagMatrix<FP>(*kvar) * wtF.transpose();
#elif defined USE_EIGEN
    cov.xy = wtF * kvar->asDiagonal() * wtF.transpose();
#endif
  }
  return TargetGalaxy<BC>( mom, cov, posn, id);
}
  
// To build the derivatives for TemplateGalaxy, we need a long
// list of coefficients for multipole terms that will go into
// the creation of each derivative of each moment.

const std::complex<double> I(0,1);

/**
struct TemplateTerm {
  int mIndex;	// Which moment it contributes to
  int dIndex;   // Which derivative it contributes to
  DComplex coeff;  // Coefficient of the term
  int m;        // Multipole number (p-q)
  int N;        // total power of k (p+q)
  TemplateTerm( int mIndex_, int dIndex_, DComplex coeff_, int m_, int N_):
    mIndex(mIndex_),
    dIndex(dIndex_),
    coeff(coeff_),
    m(m_),
    N(N_) {}
};
**/
template <class CONFIG>
std::list<typename KGalaxy<CONFIG>::TemplateTerm>
KGalaxy<CONFIG>::w0TermsEven = {
  TemplateTerm(TG::MF, TG::D0,  1.,  0, 0),
  TemplateTerm(TG::MR, TG::D0,  1.,  0, 2),
  TemplateTerm(TG::MR, TG::DU, -2.,  0, 2),
  TemplateTerm(TG::MR, TG::DV, -1., -2, 2),
  TemplateTerm(TG::MR, TG::DVb, -1., 2, 2),
  TemplateTerm(TG::MR, TG::DU_DU, 6., 0, 2),
  TemplateTerm(TG::MR, TG::DU_DV, 2., -2, 2),
  TemplateTerm(TG::MR, TG::DU_DVb, 2., 2, 2),
  TemplateTerm(TG::MR, TG::DV_DVb, 2., 0, 2),
  TemplateTerm(TG::MC, TG::D0,    1., 0, 4),
  TemplateTerm(TG::MC, TG::DU,   -4., 0, 4),
  TemplateTerm(TG::MC, TG::DV,   -2.,-2, 4),
  TemplateTerm(TG::MC, TG::DVb,  -2., 2, 4),
  TemplateTerm(TG::MC, TG::DU_DU, 20., 0, 4),
  TemplateTerm(TG::MC, TG::DU_DV,  8.,-2, 4),
  TemplateTerm(TG::MC, TG::DU_DVb, 8., 2, 4),
  TemplateTerm(TG::MC, TG::DV_DV,  2.,-4, 4),
  TemplateTerm(TG::MC, TG::DV_DVb, 6., 0, 4),
  TemplateTerm(TG::MC, TG::DVb_DVb,2., 4, 4),
  TemplateTerm(TG::ME, TG::D0,    1., 2, 2),
  TemplateTerm(TG::ME, TG::DU,   -2., 2, 2),
  TemplateTerm(TG::ME, TG::DV,   -2., 0, 2),
  TemplateTerm(TG::ME, TG::DU_DU,  6., 2, 2),
  TemplateTerm(TG::ME, TG::DU_DV,  4., 0, 2),
  TemplateTerm(TG::ME, TG::DV_DV,  2.,-2, 2),
  TemplateTerm(TG::ME, TG::DV_DVb, 1., 2, 2)
};

template <class CONFIG>
std::list<typename KGalaxy<CONFIG>::TemplateTerm>
KGalaxy<CONFIG>::w0TermsOdd = {
  TemplateTerm(TG::MX, TG::D0, I, 1, 1),
  TemplateTerm(TG::MX, TG::DU, -I, 1, 1),
  TemplateTerm(TG::MX, TG::DV, -I, -1, 1),
  TemplateTerm(TG::MX, TG::DU_DU, 2.*I, 1, 1),
  TemplateTerm(TG::MX, TG::DU_DV, I,-1, 1),
  TemplateTerm(TG::MX, TG::DV_DVb, 0.5*I, 1, 1)
};
            
// Terms using 1st deriv of W
template <class CONFIG>
std::list<typename KGalaxy<CONFIG>::TemplateTerm>
KGalaxy<CONFIG>::w1TermsEven = {
  TemplateTerm(TG::MF, TG::DU, -2.,  0, 2),
  TemplateTerm(TG::MF, TG::DV, -1., -2, 2),
  TemplateTerm(TG::MF, TG::DVb, -1., 2, 2),
  TemplateTerm(TG::MF, TG::DU_DU, 6., 0, 2),
  TemplateTerm(TG::MF, TG::DU_DV, 2.,-2, 2),
  TemplateTerm(TG::MF, TG::DU_DVb,2., 2, 2),
  TemplateTerm(TG::MF, TG::DV_DVb,2., 0, 2),
  TemplateTerm(TG::MR, TG::DU,  -2., 0, 4),
  TemplateTerm(TG::MR, TG::DV,  -1.,-2, 4),
  TemplateTerm(TG::MR, TG::DVb, -1., 2, 4),
  TemplateTerm(TG::MR, TG::DU_DU, 14., 0, 4), 
  TemplateTerm(TG::MR, TG::DU_DV,  6.,-2, 4), 
  TemplateTerm(TG::MR, TG::DU_DVb, 6., 2, 4), 
  TemplateTerm(TG::MR, TG::DV_DV , 2.,-4, 4), 
  TemplateTerm(TG::MR, TG::DV_DVb, 4., 0, 4), 
  TemplateTerm(TG::MR, TG::DVb_DVb,2., 4, 4), 
  TemplateTerm(TG::MC, TG::DU,  -2., 0, 6),
  TemplateTerm(TG::MC, TG::DV,  -1.,-2, 6),
  TemplateTerm(TG::MC, TG::DVb, -1., 2, 6),
  TemplateTerm(TG::MC, TG::DU_DU, 22., 0, 6), 
  TemplateTerm(TG::MC, TG::DU_DV, 10.,-2, 6), 
  TemplateTerm(TG::MC, TG::DU_DVb,10., 2, 6), 
  TemplateTerm(TG::MC, TG::DV_DV , 4.,-4, 6), 
  TemplateTerm(TG::MC, TG::DV_DVb, 6., 0, 6), 
  TemplateTerm(TG::MC, TG::DVb_DVb,4., 4, 6), 
  TemplateTerm(TG::ME, TG::DU,  -2., 2, 4),
  TemplateTerm(TG::ME, TG::DV,  -1., 0, 4),
  TemplateTerm(TG::ME, TG::DVb, -1., 4, 4),
  TemplateTerm(TG::ME, TG::DU_DU, 14., 2, 4), 
  TemplateTerm(TG::ME, TG::DU_DV, 10., 0, 4), 
  TemplateTerm(TG::ME, TG::DU_DVb, 4., 4, 4), 
  TemplateTerm(TG::ME, TG::DV_DV , 4.,-2, 4), 
  TemplateTerm(TG::ME, TG::DV_DVb, 4., 2, 4)
};

template <class CONFIG>
std::list<typename KGalaxy<CONFIG>::TemplateTerm>
KGalaxy<CONFIG>::w1TermsOdd = {
  TemplateTerm(TG::MX, TG::DU, -2.*I, 1, 3),
  TemplateTerm(TG::MX, TG::DV, -I,-1, 3),
  TemplateTerm(TG::MX, TG::DVb,-I, 3, 3),
  TemplateTerm(TG::MX, TG::DU_DU, 10.*I, 1, 3),
  TemplateTerm(TG::MX, TG::DU_DV,  5.*I,-1, 3),
  TemplateTerm(TG::MX, TG::DU_DVb, 3.*I, 3, 3),
  TemplateTerm(TG::MX, TG::DV_DV,  2.*I, -3, 3),
  TemplateTerm(TG::MX, TG::DV_DVb, 3.*I, 1, 3)
};

// And 2nd deriv of W
template <class CONFIG>
std::list<typename KGalaxy<CONFIG>::TemplateTerm>
KGalaxy<CONFIG>::w2TermsEven = {
  TemplateTerm(TG::MF, TG::DU_DU, 4., 0, 4),
  TemplateTerm(TG::MF, TG::DU_DV, 2.,-2, 4),
  TemplateTerm(TG::MF, TG::DU_DVb,2., 2, 4),
  TemplateTerm(TG::MF, TG::DV_DV  ,1.,-4, 4),
  TemplateTerm(TG::MF, TG::DV_DVb ,1., 0, 4),
  TemplateTerm(TG::MF, TG::DVb_DVb,1., 4, 4),
  TemplateTerm(TG::MR, TG::DU_DU, 4., 0, 6),
  TemplateTerm(TG::MR, TG::DU_DV, 2.,-2, 6),
  TemplateTerm(TG::MR, TG::DU_DVb,2., 2, 6),
  TemplateTerm(TG::MR, TG::DV_DV  ,1.,-4, 6),
  TemplateTerm(TG::MR, TG::DV_DVb ,1., 0, 6),
  TemplateTerm(TG::MR, TG::DVb_DVb,1., 4, 6),
  TemplateTerm(TG::MC, TG::DU_DU, 4., 0, 8),
  TemplateTerm(TG::MC, TG::DU_DV, 2.,-2, 8),
  TemplateTerm(TG::MC, TG::DU_DVb,2., 2, 8),
  TemplateTerm(TG::MC, TG::DV_DV  ,1.,-4, 8),
  TemplateTerm(TG::MC, TG::DV_DVb ,1., 0, 8),
  TemplateTerm(TG::MC, TG::DVb_DVb,1., 4, 8),
  TemplateTerm(TG::ME, TG::DU_DU, 4., 2, 6),
  TemplateTerm(TG::ME, TG::DU_DV, 2., 0, 6),
  TemplateTerm(TG::ME, TG::DU_DVb,2., 4, 6),
  TemplateTerm(TG::ME, TG::DV_DV  ,1.,-2, 6),
  TemplateTerm(TG::ME, TG::DV_DVb ,1., 2, 6),
  TemplateTerm(TG::ME, TG::DVb_DVb,1., 6, 6)
};

template <class CONFIG>
std::list<typename KGalaxy<CONFIG>::TemplateTerm>
KGalaxy<CONFIG>::w2TermsOdd = {
  TemplateTerm(TG::MX, TG::DU_DU,  4.*I, 1, 5),
  TemplateTerm(TG::MX, TG::DU_DV,  2.*I,-1, 5),
  TemplateTerm(TG::MX, TG::DU_DVb, 2.*I, 3, 5),
  TemplateTerm(TG::MX, TG::DV_DV,  1.*I,-3, 5),
  TemplateTerm(TG::MX, TG::DV_DVb, 1.*I, 1, 5),
  TemplateTerm(TG::MX, TG::DVb_DVb,1.*I, 5, 5)
};

template <class CONFIG>
void
KGalaxy<CONFIG>::addTerms(const std::list<TemplateTerm>& terms,
			  typename TG::DerivMatrix& destination,
			  const Vector& vw,
			  bool isEven) const {
  // Find the max order multipoles we will need for the terms
  int maxN = 0;
  int maxm = 0;
  for (auto& t : terms) {
    // Skip terms that aren't used in this configuration:
    if (t.mIndex < 0 || t.dIndex < 0) continue;
    if (abs(t.m) > maxm) maxm = abs(t.m);
    if (t.N > maxN) maxN = t.N;
  }

  // Calculate needed multipoles
  linalg::Matrix<std::complex<FP>> mN( maxm/2 + 1, maxN/2 + 1,
				       std::complex<FP>(0,0));

  // Start with m = 0 (1) for even (odd) moment calcs.
  // Only the real (imag) part of kval will contribute.
  CVector kprod(kval.size());
  int m=0;
  int N=0;
  if (isEven) {
    kprod.REAL = vw.cwiseProduct(kval.REAL);
    kprod.IMAG.setZero();
  } else {
    kprod.REAL.setZero();
    kprod.IMAG = vw.cwiseProduct(kval.IMAG);
    kprod = kprod.cwiseProduct(kz);
    m = 1;
    N = 1;
  }

  mN(m/2,N/2) = kprod.sum();
  Vector ksq = kz.cwiseProduct(kz.adjoint()).REAL;  // |k^2|
  // Proceed to higher N's
  CVector summand(kprod);
  for ( N+=2 ; N<=maxN; N+=2) {
    summand.REAL = ksq.cwiseProduct(summand.REAL);
    if (m>0) summand.IMAG =ksq.cwiseProduct(summand.IMAG);  // m=0 is pure real, can skip
    mN(m/2,N/2) = summand.sum();
  }

  // Proceed to higher m's
  for (m+=2; m<=maxm; m+=2) {
    // Multiply by 2 powers of complex k * k
    kprod = kprod.cwiseProduct(kz);
    kprod = kprod.cwiseProduct(kz);
    N = m;
    mN(m/2,N/2) = kprod.sum();

    // Proceed to higher N's
    summand = kprod;
    for ( N+=2 ; N<=maxN; N+=2) {
      summand.REAL = ksq.cwiseProduct(summand.REAL);
      summand.IMAG = ksq.cwiseProduct(summand.IMAG);
      mN(m/2,N/2) = summand.sum();
    }
  }

  // Now add in all the results dictated by the terms
  for (auto& t : terms) {
    // Skip terms that aren't used in this configuration:
    if (t.mIndex < 0 || t.dIndex < 0) continue;

    std::complex<double> z = mN(abs(t.m)/2,t.N/2);
    if (t.m >=0)
      destination(t.mIndex, t.dIndex) += t.coeff * z;
    else if (isEven)
      destination(t.mIndex, t.dIndex) += t.coeff * conj(z);
    else
      // negative odd multipole flips real-part sign.
      destination(t.mIndex, t.dIndex) -= I * t.coeff * conj(I*z);
  }
  return;
}

template <class CONFIG>
TemplateGalaxy<CONFIG>
KGalaxy<CONFIG>::getTemplate() const {
  typename TG::DerivMatrix dm(TG::MSIZE, TG::DSIZE,0.);
  typename TG::DerivMatrix dxy(TG::XYSIZE, TG::DSIZE,0.);

  // Get the weight vectors
  Vector ksq = kz.cwiseProduct(kz.adjoint()).REAL;  // |k^2|
  Matrix mw = wt->derivatives(ksq);

  // Add terms to derivs involving weight and its derivs
  Vector vw = mw.transpose().col(wt->W);  // Make sure we get a column vector for Eigen
  addTerms(w0TermsEven, dm, vw, true);
  addTerms(w0TermsOdd, dxy, vw, false);

  vw = mw.transpose().col(wt->WP);
  addTerms(w1TermsEven, dm, vw, true);
  addTerms(w1TermsOdd, dxy, vw, false);

  vw = mw.transpose().col(wt->WPP);
  addTerms(w2TermsEven, dm, vw, true);
  addTerms(w2TermsOdd, dxy, vw, false);

  FP nda = 1.;  // ????
  return TG(dm,dxy,nda,id); // ?? jSuppression ??
}

///////////////////////////////////////////////////////////////
// 
// Instantiations of the templates
//
///////////////////////////////////////////////////////////////


#define INSTANTIATE(...) \
  template class bfd::KGalaxy<BfdConfig<__VA_ARGS__>>;

#include "InstantiateMomentCases.h"

