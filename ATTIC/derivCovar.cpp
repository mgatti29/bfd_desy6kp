// Code for covariance of the moments *and derivatives* with a Gaussian weight.
// The powers of 1./sigma needed for each one are omitted, they are obtained dimensionally
// Also omitting a prefactor of (2pi)^2 n / 2 from each.

// Make a vector of all moments and their derivs.
// Moment noted by F,X,Y,P,M,S for flux, x, y, e+, ex, and size moments.
// Derivs denoted by nothing (no derivs), 1, 2, 11, 22, or 12.
// So for example S12 is d^2 M_S / dg1 dg2.

// These will be expressed in terms of the "pure moments"
// R_{Nm} + i I_{Nm} = \int d^2k \tilde I(k) k^p \bar k^q exp(-k^2 \sigma^2 /2)
// where N = p+q, m = p-q >=0.
// These pure moments have a very simple covariance matrix, in particular R_{Nm} only
// has covariance with R_{N'm} and likewise for the I's.

// The strategy is: propagate covariance matrix of the R/I "pure moments" into all
// the cov matrix for the full moments & derivatives.

#include "LinearAlgebra.h"
#include "Std.h"
#include "BinomFact.h"
#include <vector>
using namespace std;

void addCov(linalg::DMatrix& cov,
	    const std::vector<int>& ntot,
	    const std::vector<int>& ind,
	    bool mzero = false) {
  for (int i1=0; i1<ind.size(); i1++)
    for (int i2=i1; i2<ind.size(); i2++) {
      int ind1 = ind[i1];
      int ind2 = ind[i2];
      cov(ind1,ind2) = fact( (ntot[ind1]+ntot[ind2])/2) * (mzero ? 1. : 0.5);
      cov(ind2,ind1) = fact( (ntot[ind1]+ntot[ind2])/2) * (mzero ? 1. : 0.5);
    }
}

const int F = 0;
const int F1 = 1;
const int F2 = 2;
const int F11 = 3;
const int F22 = 4;
const int F12 = 5;

const int P = F + 6;
const int P1 = P+1;
const int P2 = P+2;
const int P11 = P+3;
const int P22 = P+4;
const int P12 = P+5;

const int M = P + 6;
const int M1 = M+1;
const int M2 = M+2;
const int M11 = M+3;
const int M22 = M+4;
const int M12 = M+5;

const int S = M + 6;
const int S1 = S+1;
const int S2 = S+2;
const int S11 = S+3;
const int S22 = S+4;
const int S12 = S+5;

const int X = S+6;
const int X1 = X+1;
const int X2 = X+2;
const int X11 = X+3;
const int X22 = X+4;
const int X12 = X+5;

const int Y = X+6;
const int Y1 = Y+1;
const int Y2 = Y+2;
const int Y11 = Y+3;
const int Y22 = Y+4;
const int Y12 = Y+5;

const int N = Y + 6;

// Now index the real/complex parts of the moments
// R is real, I is imag, 2 digits give N,m

const int R00 = 0;
const int R20 = 1;
const int R40 = 2;
const int R60 = 3;

const int R11 = 4;
const int I11 = 5;
const int R31 = 6;
const int I31 = 7;
const int R51 = 8;
const int I51 = 9;

const int R22 = 10;
const int I22 = 11;
const int R42 = 12;
const int I42 = 13;
const int R62 = 14;
const int I62 = 15;

const int R33 = 16;
const int I33 = 17;
const int R53 = 18;
const int I53 = 19;

const int R44 = 20;
const int I44 = 21;
const int R64 = 22;
const int I64 = 23;

const int R55 = 24;
const int I55 = 25;

const int R66 = 26;
const int I66 = 27;

const int NMOM = I66+1;

int
main(int argc, char *argv[]) {
  vector<int> ntot(NMOM,0);
  ntot[R00] = 0;
  ntot[R20] = 2;
  ntot[R40] = 4;
  ntot[R60] = 6;
  ntot[R11] = 1;
  ntot[I11] = 1;
  ntot[R31] = 3;
  ntot[I31] = 3;
  ntot[R51] = 5;
  ntot[I51] = 5;
  ntot[R22] = 2;
  ntot[I22] = 2;
  ntot[R42] = 4;
  ntot[I42] = 4;
  ntot[R62] = 6;
  ntot[I62] = 6;
  ntot[R33] = 3;
  ntot[I33] = 3;
  ntot[R53] = 5;
  ntot[I53] = 5;
  ntot[R44] = 4;
  ntot[I44] = 4;
  ntot[R64] = 6;
  ntot[I64] = 6;
  ntot[R55] = 5;
  ntot[I55] = 5;
  ntot[R66] = 6;
  ntot[I66] = 6;

  try {
    linalg::DMatrix coeff(N,NMOM,0.);
    double x = 0.5;  // This is really -dW / d (k^2) = 1/2 sigma^2, but I'm setting sigma=1.

    // Here is the guts where there are probably algebraic errors: I'm expressing each moment
    // or its derivative as a linear sum of RNm and INm moments.
    coeff(F,R00) = 1.;
    coeff(F1,R22) = 2*x;
    coeff(F2,I22) = 2*x;
    coeff(F11,R20) = -4*x;
    coeff(F11,R40) = 2*x*x;
    coeff(F11,R44) = 2*x*x;
    coeff(F22,R20) = -4*x;
    coeff(F22,R40) = 2*x*x;
    coeff(F22,R44) = -2*x*x;
    coeff(F12,I44) = 2*x*x;

    coeff(S,R20) = 1.;
    coeff(S1,R22) = -2.;
    coeff(S1,R42) = 2*x;
    coeff(S2,I22) = -2.;
    coeff(S2,I42) = 2*x;
    coeff(S11,R20) = 4.;
    coeff(S11,R40) = -8*x;
    coeff(S11,R60) = 2*x*x;
    coeff(S11,R44) = -4*x;
    coeff(S11,R64) = 2*x*x;
    coeff(S22,R20) = 4.;
    coeff(S22,R40) = -8*x;
    coeff(S22,R60) = 2*x*x;
    coeff(S22,R44) = 4*x;
    coeff(S22,R64) = -2*x*x;
    coeff(S12,I44) = -4*x;
    coeff(S12,I64) = 2*x*x;

    coeff(P,R22) = 1.;
    coeff(P1,R20) = -2.;
    coeff(P1,R40) = x;
    coeff(P1,R44) = x;
    coeff(P2,I44) = x;
    coeff(P11,R22) = 4.;
    coeff(P11,R42) = -12*x;
    coeff(P11,R62) = 3*x*x;
    coeff(P11,R66) = x*x;
    coeff(P22,R42) = -4*x;
    coeff(P22,R62) = x*x;
    coeff(P22,R66) = -x*x;
    coeff(P12,I22) = 2.;
    coeff(P12,I42) = -4*x;
    coeff(P12,I62) = x*x;
    coeff(P12,I66) = x*x;

    coeff(M,I22) = 1.;
    coeff(M1,I44) = x;
    coeff(M2,R20) = -2.;
    coeff(M2,R40) = x;
    coeff(M2,R44) = -x;
    coeff(M11,I42) = -4*x;
    coeff(M11,I62) = x*x;
    coeff(M11,I66) = x*x;
    coeff(M22,I22) = 4.;
    coeff(M22,I42) = -12*x;
    coeff(M22,I62) = 3*x*x;
    coeff(M22,I66) = -x*x;
    coeff(M12,R22) = 2.;
    coeff(M12,R42) = -4*x;
    coeff(M12,R62) = x*x;
    coeff(M12,R66) = -x*x;

    coeff(X,I11) = -1.;
    coeff(X1,I11) = 1.;
    coeff(X1,I31) = -x;
    coeff(X1,R33) = x;
    coeff(X2,R11) = -1.;
    coeff(X2,R31) = x;
    coeff(X2,I33) = x;
    coeff(X11,I11) = -1.;
    coeff(X11,I31) = 6*x;
    coeff(X11,I51) = -2*x*x;
    coeff(X11,I33) = 2*x;
    coeff(X11,I53) = -x*x;
    coeff(X11,I55) = -x*x;
    coeff(X22,I11) = -1.;
    coeff(X22,I31) = 6*x;
    coeff(X22,I51) = -2*x*x;
    coeff(X22,I33) = -2*x;
    coeff(X22,I53) = x*x;
    coeff(X22,I55) = x*x;
    coeff(X12,R33) = -2*x;
    coeff(X12,R53) = x*x;
    coeff(X12,R55) = x*x;

    coeff(Y,R11) = 1.;  
    coeff(Y1,R11) = 1.;
    coeff(Y1,R31) = -x;
    coeff(Y1,I33) = x;
    coeff(Y2,I11) = 1.;
    coeff(Y2,I31) = -x;
    coeff(Y2,R33) = -x;
    coeff(Y11,R11) = 1.;
    coeff(Y11,R31) = -6*x;
    coeff(Y11,R51) = 2*x*x;
    coeff(Y11,R33) = 2*x;
    coeff(Y11,R53) = -x*x;
    coeff(Y11,R55) = x*x;
    coeff(Y22,R11) = 1.;
    coeff(Y22,R31) = -6*x;
    coeff(Y22,R51) = 2*x*x;
    coeff(Y22,R33) = -2*x;
    coeff(Y22,R53) = x*x;
    coeff(Y22,R55) = -x*x;
    coeff(Y12,I33) = 2*x;
    coeff(Y12,I53) = -x*x;
    coeff(Y12,I55) = x*x;

    linalg::DMatrix cov(NMOM,NMOM, 0.);

    // In this section I will build the covariance matrix for the RNm/INm moments.
    // The formula are that
    // Cov(RNm, RN'm) = Cov(INm,IN'm) = (1/2) * (2 pi)^3 ((N+N')/2)! / 2 * sigma (m>0)
    // Cov(RN0, RN'0) =                         (2 pi)^3 ((N+N')/2)! / 2 * sigma (m>0).
    // I am leaving off the constant (2 pi)^3 / 2 and setting sigma=1.

    vector<int> ind;
    ind.push_back(R00);
    ind.push_back(R20);
    ind.push_back(R40);
    ind.push_back(R60);
    addCov(cov, ntot, ind, true);

    ind.clear();
    ind.push_back(R11);
    ind.push_back(R31);
    ind.push_back(R51);
    addCov(cov, ntot, ind);

    ind.clear();
    ind.push_back(R22);
    ind.push_back(R42);
    ind.push_back(R62);
    addCov(cov, ntot, ind);

    ind.clear();
    ind.push_back(R33);
    ind.push_back(R53);
    addCov(cov, ntot, ind);

    ind.clear();
    ind.push_back(R44);
    ind.push_back(R64);
    addCov(cov, ntot, ind);

    ind.clear();
    ind.push_back(R55);
    addCov(cov, ntot, ind);

    ind.clear();
    ind.push_back(R66);
    addCov(cov, ntot, ind);

    ind.clear();
    ind.push_back(I11);
    ind.push_back(I31);
    ind.push_back(I51);
    addCov(cov, ntot, ind);

    ind.clear();
    ind.push_back(I22);
    ind.push_back(I42);
    ind.push_back(I62);
    addCov(cov, ntot, ind);

    ind.clear();
    ind.push_back(I33);
    ind.push_back(I53);
    addCov(cov, ntot, ind);

    ind.clear();
    ind.push_back(I44);
    ind.push_back(I64);
    addCov(cov, ntot, ind);

    ind.clear();
    ind.push_back(I55);
    addCov(cov, ntot, ind);

    ind.clear();
    ind.push_back(I66);
    addCov(cov, ntot, ind);

#ifdef USE_TMV    
    // A necessary but not sufficient check is that the covariance matrix of the
    // R/I moments should be postive definite.
    cout << "cov SV's: " << cov.svd().getS().diag() << endl;

    // The full covariance matrix of the moments + derivs is construct here:
    linalg::DMatrix covOut = (coeff * cov) * coeff.transpose();

    // The even-moment submatrix should have 8 zero-valued SVs because it is 24 elements
    // which are linear combinations of just 16 R/I moments of even m.
    cout << "SVs for even m: " << covOut.subMatrix(0,24,0,24).svd().getS().diag() << endl;

    // The odd-moment submatrix (X and Y moments) should be pos def:
    cout << "SVs for odd m: " << covOut.subMatrix(24,36,24,36).svd().getS().diag() << endl;
#else
    cerr << "!!!!! this code needs attention for use without TMV!!!!" << endl;
#endif
  } catch (std::runtime_error& m) {
    quit(m,1);
  }
}
