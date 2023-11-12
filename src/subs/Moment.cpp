#include "Moment.h"

using namespace bfd;

////////////////////////////////////////////////
// Arithmetic
////////////////////////////////////////////////

template <class CONFIG>
Moment<CONFIG>
Moment<CONFIG>::operator+(const Moment<CONFIG>& rhs) const {
  if (BC::FixCenter)
    return Moment(m+rhs.m);
  else
    return Moment(m+rhs.m, xy+rhs.xy);
}
template <class CONFIG>
Moment<CONFIG>&
Moment<CONFIG>::operator+=(const Moment<CONFIG>& rhs) {
  m+=rhs.m;
  if (!BC::FixCenter)
    xy += rhs.xy;
  return *this;
}
template <class CONFIG>
Moment<CONFIG>
Moment<CONFIG>::operator-(const Moment<CONFIG>& rhs) const {
  if (BC::FixCenter)
    return Moment(m-rhs.m);
  else
    return Moment(m-rhs.m, xy-rhs.xy);
}
template <class CONFIG>
Moment<CONFIG>&
Moment<CONFIG>::operator-=(const Moment<CONFIG>& rhs) {
  m-=rhs.m;
  if (!BC::FixCenter)
    xy -= rhs.xy;
  return *this;
}
template <class CONFIG>
Moment<CONFIG>
Moment<CONFIG>::operator*(FP rhs) const {
  if (BC::FixCenter)
    return Moment(m*rhs);
  else
    return Moment(m*rhs, xy*rhs);
}
template <class CONFIG>
Moment<CONFIG>&
Moment<CONFIG>::operator*=(FP rhs) {
  m*=rhs;
  if (!BC::FixCenter)
    xy *= rhs;
  return *this;
}
template <class CONFIG>
Moment<CONFIG>
Moment<CONFIG>::operator/(FP rhs) const {
  if (BC::FixCenter)
    return Moment(m/rhs);
  else
    return Moment(m/rhs, xy/rhs);
}
template <class CONFIG>
Moment<CONFIG>&
Moment<CONFIG>::operator/=(FP rhs) {
  m/=rhs;
  if (!BC::FixCenter)
    xy /= rhs;
  return *this;
}


template <class CONFIG>
MomentCovariance<CONFIG>
MomentCovariance<CONFIG>::operator+(const MomentCovariance<CONFIG>& rhs) const {
  if (BC::FixCenter)
    return MomentCovariance(m+rhs.m);
  else
    return MomentCovariance(m+rhs.m, xy+rhs.xy);
}

template <class CONFIG>
MomentCovariance<CONFIG>&
MomentCovariance<CONFIG>::operator+=(const MomentCovariance<CONFIG>& rhs) {
  m+=rhs.m;
  if (!BC::FixCenter)
    xy += rhs.xy;
  return *this;
}
template <class CONFIG>
MomentCovariance<CONFIG>
MomentCovariance<CONFIG>::operator-(const MomentCovariance<CONFIG>& rhs) const {
  if (BC::FixCenter)
    return MomentCovariance(m-rhs.m);
  else
    return MomentCovariance(m-rhs.m, xy-rhs.xy);
}
template <class CONFIG>
MomentCovariance<CONFIG>&
MomentCovariance<CONFIG>::operator-=(const MomentCovariance<CONFIG>& rhs) {
  m-=rhs.m;
  if (!BC::FixCenter)
    xy -= rhs.xy;
  return *this;
}
template <class CONFIG>
MomentCovariance<CONFIG>
MomentCovariance<CONFIG>::operator*(FP rhs) const {
  if (BC::FixCenter)
    return MomentCovariance(m*rhs);
  else
    return MomentCovariance(m*rhs, xy*rhs);
}
template <class CONFIG>
MomentCovariance<CONFIG>&
MomentCovariance<CONFIG>::operator*=(FP rhs) {
  m*=rhs;
  if (!BC::FixCenter)
    xy *= rhs;
  return *this;
}
template <class CONFIG>
MomentCovariance<CONFIG>
MomentCovariance<CONFIG>::operator/(FP rhs) const {
  if (BC::FixCenter)
    return MomentCovariance(m/rhs);
  else
    return MomentCovariance(m/rhs, xy/rhs);
}
template <class CONFIG>
MomentCovariance<CONFIG>&
MomentCovariance<CONFIG>::operator/=(FP rhs) {
  m/=rhs;
  if (!BC::FixCenter)
    xy /= rhs;
  return *this;
}

////////////////////////////////////////////////
// Transformations
// All transformations assume that XY and E moments
// are the only ones that are not monopole.
////////////////////////////////////////////////

template<class CONFIG>
void Moment<CONFIG>::rotate(double theta)
{
  if (!BC::FixCenter) {
    double ncx,ncy;
    ncx=xy[BC::MX]*cos(theta)-xy[BC::MY]*sin(theta);
    ncy=xy[BC::MX]*sin(theta)+xy[BC::MY]*cos(theta);
    xy[BC::MX]=ncx;
    xy[BC::MY]=ncy;
  }

  double ne1,ne2;
  ne1=m[BC::M1]*cos(2*theta)-m[BC::M2]*sin(2*theta);
  ne2=m[BC::M1]*sin(2*theta)+m[BC::M2]*cos(2*theta);
  m[BC::M1]=ne1;
  m[BC::M2]=ne2;
  return;
}

template<class CONFIG>
void MomentCovariance<CONFIG>::rotate(double theta) {
  double c = cos(theta);
  double s = sin(theta);
  if (!BC::FixCenter) {
    // Rotate the XY covariance matrix
    typename BC::XYMatrix r;
    r(BC::MX, BC::MX) = c;
    r(BC::MX, BC::MY) = -s;
    r(BC::MY, BC::MX) = s;
    r(BC::MY, BC::MY) = c;
    // Note that SmallMatrix is not protected against aliasing
    typename BC::XYMatrix tmp = xy * r.transpose();
    xy = r * tmp;
  }

  // Now rotate the ellipticity moments
  typename BC::MMatrix r;
  r.setToIdentity();
  double c2 = c*c-s*s;
  double s2 = 2 * c * s;
  r(BC::M1, BC::M1) = c2;
  r(BC::M1, BC::M2) = -s2;
  r(BC::M2, BC::M1) = s2;
  r(BC::M2, BC::M2) = c2;
  // Note that SmallMatrix is not protected against aliasing
  typename BC::MMatrix tmp = m * r.transpose();
  m = r * tmp;
  return;
}

// Here the parity flip is defined as y -> -y
template<class CONFIG>
void Moment<CONFIG>::yflip() 
{
  if (!BC::FixCenter)
    xy[BC::MY]=-xy[BC::MY];
  m[BC::M2]=-m[BC::M2];
  return;
}

template<class CONFIG>
void MomentCovariance<CONFIG>::yflip() 
{
  if (!BC::FixCenter) {
    // We are *not* storing as SymMatrix so hit both elements
    xy(BC::MX,BC::MY) *= FP(-1);
    xy(BC::MY,BC::MX) *= FP(-1);
  }
  m.row(BC::M2) *= FP(-1);
  m.col(BC::M2) *= FP(-1);
  return;
}

template <class CONFIG>
MomentCovariance<CONFIG>
MomentCovariance<CONFIG>::isotropize() const {
  MomentCovariance<BC> out(*this);
  
  /* get rid of M1/M2 off-diagonals */
  FP m12 = 0.5*(out.m(BC::M1,BC::M1)+out.m(BC::M2,BC::M2));
  out.m.row(BC::M1) *= FP(0.);
  out.m.col(BC::M1) *= FP(0.);
  out.m.row(BC::M2) *= FP(0.);
  out.m.col(BC::M2) *= FP(0.);
  out.m(BC::M1,BC::M1) = out.m(BC::M2,BC::M2) = m12;

  /* Isotropize centroid moments */
  if (!BC::FixCenter) {
    out.xy(BC::MX,BC::MY) = out.xy(BC::MY,BC::MX) = FP(0.);
    FP varxy = 0.5*(out.xy(BC::MX,BC::MX)+out.xy(BC::MY,BC::MY));
    out.xy(BC::MX,BC::MX) = out.xy(BC::MY,BC::MY) = varxy;
  }
  return out;
}


template <class CONFIG>
bool
MomentCovariance<CONFIG>::isIsotropic(double tolerance) const {
  auto iso = isotropize();

  // The difference between input cov and the isotropized version should be
  // < tol times diag elements of cov

  typename BC::MVector invsig;	// get diagonal element scalings
  for (int i=0; i<BC::MSIZE; i++) invsig[i] = 1./sqrt(iso.m(i,i));
  typename BC::MMatrix dm = invsig.asDiagonal() * (m - iso.m) * invsig.asDiagonal();

  // Check that all elements are small
  if (dm.cwiseAbs().maxCoeff() > tolerance)
    return false;

  // Same thing for the XY matrix
  if (!BC::FixCenter) {
    FP varx = iso.xy(BC::MX,BC::MX);
    if ((xy-iso.xy).cwiseAbs().maxCoeff() > tolerance*varx)
      return false;
  }
  return true;
}

///////////////////////////////////////////////////////////////
// 
// Instantiations of the templates
//
///////////////////////////////////////////////////////////////

#define INSTANTIATE(...)	\
  template class bfd::Moment<BfdConfig<__VA_ARGS__> >;		\
  template class bfd::MomentCovariance<BfdConfig<__VA_ARGS__> >;

#include "InstantiateMomentCases.h"
