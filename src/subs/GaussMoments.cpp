// Code to calculate moments for Gaussians.
#include "GaussMoments.h"
#include <BinomFact.h>

using namespace bfd;

void
GaussMoments::setY() const {
  double k0sq = 1./(sigsq + sigsqW);
  double eR = e * sigsq/(sigsq+sigsqW);
  double s = pow(1 - eR*eR , -0.5);
  // powers of s:
  linalg::DVector sN(pqMax+2);
  sN[0] = 1.;
  for (int i=1; i<sN.size(); i++) sN[i] = sN[i-1]*s;

  Y.setZero();

  // Do m=0:
  double Z = 2 * PI * k0sq;
  Y(0,0) = Z * s;
  Z *= 2*k0sq;
  Y(1,1) = Z * sN[3];
  Z *= 2*k0sq;
  Y(2,2) = Z * (3*sN[5]-sN[3]);
  Z *= 2*k0sq;
  Y(3,3) = Z * (15*sN[7]-9*sN[5]);
  Z *= 2*k0sq;
  Y(4,4) = Z * (105*sN[9]-90*sN[7]+9*sN[5]);
  
  // m=1 * 2:
  Z = -4*PI*eR * k0sq*k0sq;
  Y(2,0) = Z * sN[3];
  Z *= 2*k0sq;
  Y(3,1) = Z * (3*sN[5]);
  Z *= 2*k0sq;
  Y(4,2) = Z * (15*sN[7]-3*sN[5]);
  Z *= 2*k0sq;
  Y(5,3) = Z * (105*sN[9]-45*sN[7]);

  // m=2 * 2:
  Z = 24*PI* eR*eR * pow(k0sq,3.);
  Y(4,0) = Z * sN[5];
  Z *= 2*k0sq;
  Y(5,1) = Z * (5*sN[7]);
  Z *= 2*k0sq;
  Y(6,2) = Z * (35*sN[9]-5*sN[7]);

  // m=3 * 2:
  Z = -240*PI*pow(eR, 3.) * pow(k0sq,4.);
  Y(6,0) = Z*sN[7];
  Z *= 2*k0sq;
  Y(7,1) = Z * (7*sN[9]);

  // m=4 * 2:
  Z = 3360*PI*pow(eR, 4.) * pow(k0sq,5.);
  Y(8,0) = Z * sN[9];

  // Fill upper half of Y matrix, for safety
  for (int i=0; i<Y.nrows(); i++)
    for (int j=0; j<i; j++)
      Y(j,i) = Y(i,j);
  
  return;
}

DComplex
GaussMoments::moment(int p, int q) const {
  if (p+q > pqMax)
    FormatAndThrow<std::runtime_error>() << "(p,q)=("
					     << p << "," << q
					     << ") exceeds maximumum order for GaussMoments";
  DComplex phase( cos((p-q)*beta), sin((p-q)*beta));
  if (real(u)==0. && imag(u)==0.) return flux * phase * Y(p,q);
  DComplex I(0.,1.);
  DComplex out = 0.;
  DComplex uj = 1.; // (iu)^j
  // Note negative signs are needed on u in the u^p ubar^q terms to correct
  // an error in the FFT sign convention in Gary's 2/26/14 notes.
  for (int j=0; j<=p; uj*=-u*I, j++) {
    DComplex factor = uj * binom(p, j);
    DComplex ubark = 1.; // \bar(iu)^k
    for (int k=0; k<=q; ubark*=-I*conj(u), k++) {
      if ( (p+q-j-k)%2 !=0) continue; // Only even moments are nonzero
      out += factor * binom(q,k) * ubark * Y(p-j,q-k);
    }
  }
  out *= flux * phase * x0Suppression;
  return out;
}

void
GaussMoments::setX0(double x0_, double y0_) {
  x0 = x0_;
  y0 = y0_;
  setU();
}

void
GaussMoments::setBeta(double beta_) {
  beta = beta_;
  setU();
}

void
GaussMoments::setU() {
  // Rotate centroid shift into principle axes
  DComplex xprime = DComplex( cos(beta), -sin(beta)) * DComplex(x0, y0);
  u = DComplex( real(xprime)/(sigmax*sigmax), imag(xprime)/(sigmay*sigmay) );
  x0Suppression = exp(-0.5* (real(u)*real(u)*sigmax*sigmax + imag(u)*imag(u)*sigmay*sigmay));
}

