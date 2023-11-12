// Code to get the moments and derivatives from pixelized data.

#include "PixelGalaxy.h"
#include "UseTMV.h"
#include "fft.h"

using namespace bfd;
using tmv::ElemProd;

KData
KData::getShifted(double dx, double dy) {
  // Note that dx,dy are shift of coordinate origin so phase is e^ikx.
  for (int i=0; i<re.size(); i++) {
    double phi = dx * kx[i] + dy*ky[i];
    DComplex z = DComplex(re[i],im[i]) * DComplex(cos(phi),sin(phi));
    re[i] = z.real();
    im[i] = z.imag();
  }
}

template<int UseMoments>
WKMoments<UseMoments>::WKMoments(const WKDerivs<UseMoments>& wkd):
  evens(wkd.evens.colRange(0,N_EVEN)),
  odds(wkd.odds.colRange(0,N_ODD)) {}


template<int UseMoments>
WKMoments<UseMoments>::WKMoments(const KWeight& kw, const KData& kd) {
  Assert(kd.kx.size() == kd.ky.size());
  int nk = kd.kx.size();

  if (N_EVEN > 0) {
    evens.resize(nk, N_EVEN);
    evens.setZero();
  }
  if (N_ODD > 0) {
    odds.resize(nk, N_ODD);
    odds.setZero();
  }

  DVector ksq = ElemProd(kd.kx,kd.kx) + ElemProd(kd.ky,kd.ky);
  DVector w = kw(ksq);

  w *= 2. * kd.d2k;

  if (MI::UseFlux) {
    evens.col(FLUX) = w;
    evens(kd.dcIndex, FLUX) *= 0.5;
  }
  if (MI::UseSize) {
    evens.col(SIZE) = ElemProd(w,ksq);
  }
  if (MI::UseCentroid) {
    odds.col(CX) = ElemProd(-w,kd.kx);
    odds.col(CY) = ElemProd(-w,kd.ky);
  }
  if (MI::UseE) {
    DVector k2x = ElemProd(kd.kx,kd.kx) - ElemProd(kd.ky,kd.ky);
    DVector k2y = ElemProd(2. * kd.kx,kd.ky);
    evens.col(E1) = ElemProd(w,k2x);
    evens.col(E2) = ElemProd(w,k2y);
  }
}

template<int UseMoments>
Moments<UseMoments> 
WKMoments<UseMoments>::getMoments(const KData& kd) const {
  Moments<UseMoments> out;
  if (N_EVEN > 0) {
    DVector e = kd.re * evens;
    if (MI::UseFlux) out[MI::FLUX] = e[FLUX];
    if (MI::UseSize) out[MI::SIZE] = e[SIZE];
    if (MI::UseE) {
      out[MI::E1] = e[E1];
      out[MI::E2] = e[E2];
    }
  }
  if (N_ODD > 0) {
    Assert(MI::UseCentroid);
    DVector o = kd.im * odds;
    out[MI::CX] = o[CX];
    out[MI::CY] = o[CY];
  }
  return out;
}

template<int UseMoments>
Moments<UseMoments> 
WKMoments<UseMoments>::dMdx(const KData& kd) const {
  Moments<UseMoments> out;
  if (N_EVEN > 0) {
    DVector e = ElemProd(kd.kx, kd.im) * evens;
    if (MI::UseFlux) out[MI::FLUX] = e[FLUX];
    if (MI::UseSize) out[MI::SIZE] = e[SIZE];
    if (MI::UseE) {
      out[MI::E1] = e[E1];
      out[MI::E2] = e[E2];
    }
  }
  if (N_ODD > 0) {
    Assert(MI::UseCentroid);
    DVector o = ElemProd(kd.kx, kd.re) * odds;
    out[MI::CX] = -o[CX];
    out[MI::CY] = -o[CY];
  }
  return out;
}

template<int UseMoments>
Moments<UseMoments> 
WKMoments<UseMoments>::dMdy(const KData& kd) const {
  Moments<UseMoments> out;
  if (N_EVEN > 0) {
    DVector e = ElemProd(kd.ky, kd.im) * evens;
    if (MI::UseFlux) out[MI::FLUX] = e[FLUX];
    if (MI::UseSize) out[MI::SIZE] = e[SIZE];
    if (MI::UseE) {
      out[MI::E1] = e[E1];
      out[MI::E2] = e[E2];
    }
  }
  if (N_ODD > 0) {
    Assert(MI::UseCentroid);
    DVector o = ElemProd(kd.ky, kd.re) * odds;
    out[MI::CX] = -o[CX];
    out[MI::CY] = -o[CY];
  }
  return out;
}

template<int UseMoments>
WKDerivs<UseMoments>::WKDerivs(const KWeight& kw, const KData& kd) {
  Assert(kd.kx.size() == kd.ky.size());
  int nk = kd.kx.size();

  if (N_EVEN > 0) {
    evens.resize(nk, N_EVEN);
    evens.setZero();
  }
  if (N_ODD > 0) {
    odds.resize(nk, N_ODD);
    odds.setZero();
  }

  DVector ksq = ElemProd(kd.kx,kd.kx) + ElemProd(kd.ky,kd.ky);
  DMatrix wderivs = kw.derivatives(ksq);
  wderivs *= 2. * kd.d2k;
  DVector w = wderivs.col(0);
  DVector wp = wderivs.col(1);
  DVector wpp = wderivs.col(2);

  // We will collect all our contributions in ascending m order
  // kpX, kpY will hold components of k^p
  // Inner loop to raise p,q by 1 each to give kpqX, kpxY

  DVector temp(nk);

  // m=0 terms: 
  // pq = 00
  if (MI::UseFlux) {
    evens.col(FLUX) = w;
    evens(kd.dcIndex, FLUX) *= 0.5;
  }
  // pq=11
  DVector kpqX = ksq;
  DVector kpqY(nk);
  if (MI::UseSize) evens.col(SIZE)      = ElemProd(w,ksq);
  if (MI::UseE)    evens.col(M2_V_REAL) = ElemProd(-2.*w,ksq);
  if (MI::UseSize) evens.col(MR_I2)= ElemProd(4.*w,ksq);
  if (MI::UseFlux) evens.col(M0_I2)= ElemProd(4.*wp,ksq);

  // pq=22
  kpqX = ElemProd(kpqX,ksq);
  if (MI::UseSize) evens.col(MR_I2)  += ElemProd(8.*wp,kpqX);
  if (MI::UseE)    evens.col(M2_V_REAL)   += ElemProd(-wp,kpqX);
  if (MI::UseFlux) evens.col(M0_I2)  += ElemProd(2.*wpp,kpqX);
  
  // pq=33
  if (MI::UseSize) {
    kpqX = ElemProd(kpqX, ksq);
    evens.col(MR_I2) += ElemProd(2.*wpp, kpqX);
  }

  // m=1 terms
  DVector kpX = kd.kx;
  DVector kpY = kd.ky;
  if (MI::UseCentroid) {
    // pq = 10
    odds.col(CX) = ElemProd(-w,kpX);
    odds.col(CY) = ElemProd(-w,kpY);
    odds.col(M1_V_REAL) = -odds.col(CX);
    odds.col(M1_V_IMAG) =  odds.col(CY);
    odds.col(M1_I2_REAL) = odds.col(CX);
    odds.col(M1_I2_IMAG) = odds.col(CY);

    // pq = 21
    kpqX = ElemProd(kpX, ksq);
    kpqY = ElemProd(kpY, ksq);
    temp = ElemProd( wp, kpqX);
    odds.col(M1_V_REAL) += temp;
    odds.col(M1_I2_REAL) += -6. * temp;

    temp = ElemProd( -wp, kpqY);
    odds.col(M1_V_IMAG) += temp;
    odds.col(M1_I2_IMAG) +=  6.*temp;
    
    // pq = 32
    kpqX = ElemProd(kpqX, ksq);
    kpqY = ElemProd(kpqY, ksq);
    odds.col(M1_I2_REAL) += ElemProd( -2.*wpp, kpqX);
    odds.col(M1_I2_IMAG) += ElemProd( -2.*wpp, kpqY);
  }

  // m=2:
  temp = ElemProd(kpX, kd.kx) - ElemProd(kpY, kd.ky);
  kpY  = ElemProd(kpX, kd.ky) + ElemProd(kpY, kd.kx);
  kpX = temp;

  // pq=20
  if (MI::UseE) {
    evens.col(E1) = ElemProd(w,kpX);
    evens.col(E2) = ElemProd(w,kpY);
    evens.col(M2_I2_REAL)  =  2.*evens.col(E1);
    evens.col(M2_I2_IMAG)  =  2.*evens.col(E2);
    evens.col(M2_VV_REAL)  =  2.*evens.col(E1);
    evens.col(M2_VV_IMAG)  = -2.*evens.col(E2);
  }
  if (MI::UseSize) {
    evens.col(MR_V_REAL) = ElemProd(-w, kpX);
    evens.col(MR_V_IMAG) = ElemProd( w, kpY);
  }
  if (MI::UseFlux) {
    evens.col(M0_V_REAL) = ElemProd(-wp, kpX);
    evens.col(M0_V_IMAG) = ElemProd( wp, kpY);
  }
  
  // pq = 31
  kpqX = ElemProd(kpX, ksq);
  kpqY = ElemProd(kpY, ksq);
  if (MI::UseSize) {
    evens.col(MR_V_REAL) += ElemProd(-wp, kpqX);
    evens.col(MR_V_IMAG) += ElemProd( wp, kpqY);
  }
  if (MI::UseE) {
    temp = ElemProd(wp, kpqX);
    evens.col(M2_I2_REAL) += 8.*temp;
    evens.col(M2_VV_REAL) += 4.*temp;
    temp = ElemProd(wp, kpqY);
    evens.col(M2_I2_IMAG) += 8.*temp;
    evens.col(M2_VV_IMAG) += -4.*temp;

    // pq = 42
    kpqX = ElemProd(kpqX, ksq);
    kpqY = ElemProd(kpqY, ksq);
    temp = ElemProd(wpp, kpqX);
    evens.col(M2_I2_REAL) += 2.*temp;
    evens.col(M2_VV_REAL) += temp;
    temp = ElemProd(wpp, kpqY);
    evens.col(M2_I2_IMAG) += 2.*temp;
    evens.col(M2_VV_IMAG) += -temp;
  }

  // m=3
  temp = ElemProd(kpX, kd.kx) - ElemProd(kpY, kd.ky);
  kpY  = ElemProd(kpX, kd.ky) + ElemProd(kpY, kd.kx);
  kpX = temp;

  if (MI::UseCentroid) {
    // pq = 30
    odds.col(M1_VB_REAL) = ElemProd(wp, kpX);
    odds.col(M1_VB_IMAG) = ElemProd(wp, kpY);
    odds.col(M1_VV_REAL) = -2. * odds.col(M1_VB_REAL);
    odds.col(M1_VV_IMAG) =  2. * odds.col(M1_VB_IMAG);

    // pq = 41
    kpqX = ElemProd(kpX, ksq);
    kpqY = ElemProd(kpY, ksq);
    odds.col(M1_VV_REAL) += ElemProd(-wpp, kpqX);
    odds.col(M1_VV_IMAG) += ElemProd( wpp, kpqY);
  }

  // m=4
  temp = ElemProd(kpX, kd.kx) - ElemProd(kpY, kd.ky);
  kpY  = ElemProd(kpX, kd.ky) + ElemProd(kpY, kd.kx);
  kpX = temp;

  // pq=40
  if (MI::UseE) {
    evens.col(M2_VB_REAL) = ElemProd(-wp, kpX);
    evens.col(M2_VB_IMAG) = ElemProd(-wp, kpY);
  }
  if (MI::UseSize) {
    evens.col(MR_VV_REAL) = ElemProd( 2.*wp, kpX);
    evens.col(MR_VV_IMAG) = ElemProd(-2.*wp, kpY);
  }
  if (MI::UseFlux) {
    evens.col(M0_VV_REAL) = ElemProd( wpp, kpX);
    evens.col(M0_VV_IMAG) = ElemProd(-wpp, kpY);
  }

  // pq=51
  if (MI::UseSize) {
    kpqX = ElemProd(kpX, ksq);
    kpqY = ElemProd(kpY, ksq);
    evens.col(MR_VV_REAL) += ElemProd( wpp, kpqX);
    evens.col(MR_VV_IMAG) += ElemProd(-wpp, kpqY);
  }

  // m = 5
  if (MI::UseCentroid || MI::UseE) {
    temp = ElemProd(kpX, kd.kx) - ElemProd(kpY, kd.ky);
    kpY  = ElemProd(kpX, kd.ky) + ElemProd(kpY, kd.kx);
    kpX = temp;
  }
  // pq = 50
  if (MI::UseCentroid) {
    odds.col(M1_VVB_REAL) = ElemProd(-wpp, kpX);
    odds.col(M1_VVB_IMAG) = ElemProd(-wpp, kpY);
  }

  // m=6
  if (MI::UseE) {
    temp = ElemProd(kpX, kd.kx) - ElemProd(kpY, kd.ky);
    kpY  = ElemProd(kpX, kd.ky) + ElemProd(kpY, kd.kx);
    kpX = temp;
    // pq = 60
    evens.col(M2_VVB_REAL) = ElemProd(wpp, kpX);
    evens.col(M2_VVB_IMAG) = ElemProd(wpp, kpY);
  }
}

template<int UseMoments>
MomentDecomposition<UseMoments> 
WKDerivs<UseMoments>::getDecomposition(const KData& kd) const {
  MomentDecomposition<UseMoments> out;
  if (N_EVEN > 0) {
    DVector e = kd.re * evens;
    if (MI::UseFlux) {
      out.f[MI::FLUX] = e[FLUX];
      out.dv[MI::FLUX] = DComplex( e[M0_V_REAL], e[M0_V_IMAG] );
      out.d2I[MI::FLUX] = e[M0_I2];
      out.d2vv[MI::FLUX] = DComplex( e[M0_VV_REAL], e[M0_VV_IMAG] );
    }
    if (MI::UseSize) {
      out.f[MI::SIZE] = e[SIZE];
      out.dv[MI::SIZE] = DComplex( e[MR_V_REAL], e[MR_V_IMAG] );
      out.d2I[MI::SIZE] = e[MR_I2];
      out.d2vv[MI::SIZE] = DComplex( e[MR_VV_REAL], e[MR_VV_IMAG] );
    }
    if (MI::UseE) {
      out.f[MI::E1] = e[E1];
      out.f[MI::E2] = e[E2];
      out.dv[MI::E1] = DComplex( e[M2_V_REAL], e[M2_V_IMAG] );
      out.dvbar[MI::E1] = DComplex( e[M2_VB_REAL], e[M2_VB_IMAG] );
      out.d2I[MI::E1] = DComplex( e[M2_I2_REAL], e[M2_I2_IMAG] );
      out.d2vv[MI::E1] = DComplex( e[M2_VV_REAL], e[M2_VV_IMAG] );
      out.d2vvbar[MI::E1] = DComplex( e[M2_VVB_REAL], e[M2_VVB_IMAG] );
    }
  }
  if (N_ODD > 0) {
    Assert(MI::UseCentroid);
    DVector o = kd.im * odds;
    out.f[MI::CX] = o[CX];
    out.f[MI::CY] = o[CY];
    out.dv[MI::CX] = DComplex( o[M1_V_REAL], o[M1_V_IMAG] );
    out.dvbar[MI::CX] = DComplex( o[M1_VB_REAL], o[M1_VB_IMAG] );
    out.d2I[MI::CX] = DComplex( o[M1_I2_REAL], o[M1_I2_IMAG] );
    out.d2vv[MI::CX] = DComplex( o[M1_VV_REAL], o[M1_VV_IMAG] );
    out.d2vvbar[MI::CX] = DComplex( o[M1_VVB_REAL], o[M1_VVB_IMAG] );
  }
  return out;
}

///////////////////////////////////////////////////////////////
// Making covariance matrices
///////////////////////////////////////////////////////////////

template<int UseMoments>
MomentCovariance<UseMoments> 
WKMoments<UseMoments>::getCov(const KData& psf) const {
  Assert(psf.kx.size()==evens.colsize());
  const int nk = psf.kx.size();

  
  // Make a copy of the coefficient matrix, divided by |T^2| at each k:
  // (could do this with a diagonal matrix)
  DMatrix evens2 = evens;
  for (int i=0; i<nk; i++)
    evens2.row(i) *= 1. / (psf.re[i]*psf.re[i]+psf.im[i]*psf.im[i]);
  evens2.row(psf.dcIndex) *= 2.;   // DC component has 2x higher variance

  // Use TMV's routine for multiplying two matrices when you 
  // know the result is symmetric: ecov = evens^T * evens2 * scalar;
  tmv::SymMatrix<double> ecov(N_EVEN);
  double scalar = 0.5 * 4*PI*PI / psf.d2k;
  tmv::SymMultMM<false>( scalar , evens.transpose(), evens2, ecov.view());

  // Slot the covariance elements into the MomentCovariance matrix
  MomentCovariance<UseMoments> out;
  if (MI::UseFlux) {
    out(MI::FLUX, MI::FLUX) = ecov(FLUX,FLUX);
    if (MI::UseSize) 
      out(MI::FLUX, MI::SIZE) = out(MI::SIZE, MI::FLUX) = ecov(FLUX,SIZE);
    if (MI::UseE) {
      out(MI::FLUX, MI::E1) = out(MI::E1, MI::FLUX) = ecov(FLUX,E1);
      out(MI::FLUX, MI::E2) = out(MI::E2, MI::FLUX) = ecov(FLUX,E2);
    }
  }
  if (MI::UseSize) {
      out(MI::SIZE, MI::SIZE) = ecov(SIZE,SIZE);
    if (MI::UseE) {
      out(MI::SIZE, MI::E1) = out(MI::E1, MI::SIZE) = ecov(SIZE,E1);
      out(MI::SIZE, MI::E2) = out(MI::E2, MI::SIZE) = ecov(SIZE,E2);
    }
  }
  if (MI::UseE) {
      out(MI::E1, MI::E1) = ecov(E1,E1);
      out(MI::E1, MI::E2) = out(MI::E2, MI::E1) = ecov(E1,E2);
      out(MI::E2, MI::E2) = ecov(E2,E2);
  }

  // Do the odds if there is centroid measurement:
  if (MI::UseCentroid) {
    Assert(N_ODD > 0);
    DMatrix odds2 = odds;
    for (int i=0; i<nk; i++)
      odds2.row(i) *= 1. / (psf.re[i]*psf.re[i]+psf.im[i]*psf.im[i]);

    // Use TMV's routine for multiplying two matrices when you 
    // know the result is symmetric: ocov = odds^T * odds2 * scalar
    tmv::SymMatrix<double> ocov(N_ODD);
    tmv::SymMultMM<false>( scalar , odds.transpose(), odds2, ocov.view());
    out(MI::CX, MI::CX) = ocov(CX,CX);
    out(MI::CX, MI::CY) = out(MI::CY, MI::CX) = ocov(CX,CY);
    out(MI::CY, MI::CY) = ocov(CY,CY);
  }
  return out;
}

///////////////////////////////////////////////////////////////
// 
// PixelGalaxy routines to extract moments 
//
///////////////////////////////////////////////////////////////

template<int UseMoments>
Moments<UseMoments> 
PixelGalaxy<UseMoments>::getMoments() const {
  return getWKMoments()->getMoments(kd);
}

template<int UseMoments>
const WKMoments<UseMoments>*
PixelGalaxy<UseMoments>::getWKMoments() const {
  if (wkm) return wkm;
  else if (wkd) return new WKMoments<UseMoments>(*wkd);
  else {
    wkm = new WKMoments<UseMoments>(Galaxy<UseMoments>::kw, kd);
    return wkm;
  }
}

template<int UseMoments>
void
PixelGalaxy<UseMoments>::copyWKMoments(const WKMoments<UseMoments>& wkmIn) const {
  if (wkm) delete wkm;
  wkm = new WKMoments<UseMoments>(wkmIn);
}

template<int UseMoments>
MomentDecomposition<UseMoments> 
PixelGalaxy<UseMoments>::getDecomposition() const {
  return getWKDerivs()->getDecomposition(kd);
}

template<int UseMoments>
const WKDerivs<UseMoments>*
PixelGalaxy<UseMoments>::getWKDerivs() const {
  if (!wkd) wkd = new WKDerivs<UseMoments>(Galaxy<UseMoments>::kw,kd);
  return wkd;
}

template<int UseMoments>
void
PixelGalaxy<UseMoments>::copyWKDerivs(const WKDerivs<UseMoments>& wkdIn) const {
  if (wkd) delete wkd;
  wkd = new WKDerivs<UseMoments>(wkdIn);
}

template<int UseMoments>
MomentCovariance<UseMoments>
PixelGalaxy<UseMoments>::getCov() const {
  MomentCovariance<UseMoments> out = getWKMoments()->getCov(psfData);
  out *= noise;
  return out;
}

template<int UseMoments>
TemplateGalaxy<UseMoments>*
PixelGalaxy<UseMoments>::getShifted(double dx, double dy) const {
  // Make a new copy of this galaxy.  The wkm / wkd coefficients
  // will be unaltered by this shift, so keep them.
  PixelGalaxy<UseMoments>* out = new PixelGalaxy<UseMoments>(*this);
  // Then just apply phases to the kdata:
  out->kd.shift(dx, dy);
  return out;
}

///////////////////////////////////////////////////////////////
// 
// PixelGalaxy constructor that gets KData from an image
//
///////////////////////////////////////////////////////////////

template<int UseMoments>
PixelGalaxy<UseMoments>::PixelGalaxy(const KWeight& kw_, 
				     const img::Image<> img,
				     const Affine& map,
				     const Psf& psf, 
				     double noise_): TemplateGalaxy<UseMoments>(kw_),
  noise(noise_), wkm(0), wkd(0) {
  // ??? Check that psf kmax >= kw.kMax()
  // ??? check for aliasing

  // Assume that the FFT has to span the postage stamp given as input.
  int Nimg = MAX( img.yMax() - img.yMin(), img.xMax() - img.xMin()) + 1;
  int Nfft = fft::goodFFTSize(Nimg);

  // Ideally the coordinate origin is between (-1,-1) and (0,0) when
  // the FFT is numbered from -Nfft/2 -> Nfft/2 - 1
  // The pixel at (xMin,yMin) will be put into (-Nfft/2, -Nfft/2) of the FFT.

  double xOrigin = map.getX0();
  int xMin = static_cast<int> (ceil(xOrigin)) - Nfft/2;
  // But must include all the given data:
  xMin = MIN(xMin, img.xMin());
  xMin = MAX(xMin, img.xMax() - Nfft + 1);
  Assert( xMin <= img.xMin() && xMin+Nfft-1 >= img.xMax());

  double yOrigin = map.getY0();
  int yMin = static_cast<int> (ceil(yOrigin)) - Nfft/2;
  // But must include all the given data:
  yMin = MIN(yMin, img.yMin());
  yMin = MAX(yMin, img.yMax() - Nfft + 1);
  Assert( yMin <= img.yMin() && yMin+Nfft-1 >= img.yMax());

  // [xy]Min are now the pixel coordinates of the first pixel to
  // be included in the FFT.  Will have index -Nfft/2 in the FFT coords.
  // Could be outside the input image.

  // These are (FFT coordinate - pixel coordinate)
  int xPixelShift = -Nfft/2 - xMin;
  int yPixelShift = -Nfft/2 - yMin;
  // These are the FFT coordinates of the phase center
  double xPhaseCtr = xOrigin + xPixelShift;
  double yPhaseCtr = yOrigin + yPixelShift;

  // ???? Issue warning if the phase shift is going to be too large.

  // Copy image data into FFT array
  fft::XTable xt(Nfft, 1., 0.);
  for (int iy = img.yMin(); iy<=img.yMax(); iy++) 
    for (int ix = img.xMin(); ix<=img.xMax(); ix++) 
      xt.xSet(ix+xPixelShift, iy+yPixelShift, img(ix,iy));

  // Do the FT:
  fft::KTable* kt = xt.transform();
  const double dk = kt->getDk();
  // Shift phase center to input coordinates
  /**/ try {
    kt->translate(-xPhaseCtr, -yPhaseCtr);
  } catch (fft::FFTOutofRange& e) {
    cerr << "fft::translate() exception with phaseCtr (" << xPhaseCtr << "," << yPhaseCtr << endl;
    cerr << "Origin " << xOrigin << ", " << yOrigin << endl;
    cerr << "xyMin " << xMin << ", " << yMin << endl;
    cerr << "Pixel shift " << xPixelShift << ", " << yPixelShift << endl;
    throw;
  }

  // Get k grid size in sky coordinates
  double detA = map.jacobian();
  double d2k = dk * dk / detA;
  // Rescale noise density into sky units:
  noise /= detA;

  // Make temporary vectors to hold useful points
  vector<double> re;
  vector<double> im;
  vector<double> kx;
  vector<double> ky;
  vector<double> psfre;
  vector<double> psfim;

  // Fill ky>=0:
  DVector2 kpix;
  double kmaxsq = pow(Galaxy<UseMoments>::kw.kMax(),2.);
  int dcIndex = -1;

  for (int iy=-Nfft/2+1; iy<=Nfft/2; iy++) {
    kpix[1] = iy * dk;
    // Do not duplicate the points at ix=0, Nfft/2 in iy<0 half-plane.
    // Their conjugates were already counted in the upper half-plane.
    int ixStart = (iy>=0) ? 0 : 1;
    int ixEnd =  (iy>=0) ? Nfft/2 : Nfft/2-1;
    for (int ix=ixStart; ix<=ixEnd; ix++) {
      kpix[0] = ix*dk;
      DVector2 ksky = map.kFwd(kpix);
      if ( ksky[0]*ksky[0]+ksky[1]*ksky[1] >= kmaxsq) continue;
      if (iy==0 && ix==0) dcIndex = kx.size();

      DComplex mtf = psf.kValue(ksky[0],ksky[1]);
      // This is the deconvolved FT of the image!!!:
      DComplex z = kt->kval(ix,iy) / mtf;
      
      // For the k values that are around the edges of the kTable
      // (at coordinates Nfft/2), we're going to put half of the
      // coefficient at positive ky and half at negative ky so as
      // to symmetrize the k integrals.
      // This should not matter if we properly have the weight function
      // going to zero by kmax of the grid.  But I insert this
      // to make some test cases (like Gaussian weights)
      // work a little better
      bool flipY = false;
      if (ix==Nfft/2) {
	if (iy==Nfft/2) {
	  // The corner point at ix=iy=Nfft/2
	  // is also self-conjugate, extra factor 1/2
	  flipY = true;
	  z *= 0.25;
	} else if (iy>0) {
	  flipY = true;
	  z *= 0.5;
	}
      } else if (iy==Nfft/2 && ix>0) {
	flipY = true;
	z *= 0.5;
      }
     
      re.push_back(z.real());
      im.push_back(z.imag());
      psfre.push_back(mtf.real());
      psfim.push_back(mtf.imag());
      kx.push_back(ksky[0]);
      ky.push_back(ksky[1]);

      if (flipY) {
	re.push_back(z.real());
	im.push_back(z.imag());
	psfre.push_back(mtf.real());
	psfim.push_back(mtf.imag());
	kx.push_back(ksky[0]);
	ky.push_back(-ksky[1]);
      }
    }
  }
  delete kt;

  /**cerr << "PixelGalaxy dk " << dk/sqrt(detA)
	   << " Nfft " << Nfft
	   << " kept " << re.size()
	   << " kmaxsq " << kmaxsq
	   << endl;
  /**/
  // Save the results in our KData
  kd.set(re, im, kx, ky, d2k, dcIndex);
  psfData.set(psfre, psfim, kx, ky, d2k, dcIndex);
}

template<int UseMoments>
PixelGalaxy<UseMoments>::~PixelGalaxy() {
  if (wkd) delete wkd;
  if (wkm) delete wkm;
}

///////////////////////////////////////////////////////////////
// 
// Instantiations of the templates
//
///////////////////////////////////////////////////////////////


#define INSTANTIATE(u) \
  template class WKMoments<u>; \
  template class WKDerivs<u>; \
  template class PixelGalaxy<u>;

#include "InstantiateMomentCases.h"

