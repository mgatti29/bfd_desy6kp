// Code that compares the analytic GaussMoments outputs to a numerical integration.
#include "Std.h"
#include "GaussMoments.h"
#include "LinearAlgebra.h"

using namespace bfd;
using namespace linalg;

const int maxN = 8;	// Highest p+q moment

// Return matrix of powers k^p bar k^q
CMatrix kpow(DComplex k) {
  CMatrix out(maxN+1, maxN+1, 0.);
  DComplex kp = 1.;
  for (int p=0; p<=maxN; kp*=k, ++p) {
    DComplex kpq = kp;
    for (int q=0; q<=p && p+q<=maxN; kpq *= conj(k), ++q) {
      out(p,q) = kpq;
      out(q,p) = conj(kpq);
    }
  }
  return out;
}

int main(int argc,
	 char *argv[])
{

  try {
    double flux = 1.;
    double sigma = 1.5;
    double e = 0.;
    double sigmaW = 1.;
    double beta = 0.;
    double x0 = 0.;
    double y0 = 0.;

    if (argc<2 || argc > 8) {
      // Use default set of Gaussian parameters
      flux = 2.;
      sigma = 3.;
      e = 0.7;
      sigmaW = 0.5;
      beta = 40. * 3.1415/180.;
      x0 = 0.1;
      y0 = -1.2;
      cerr 
	<< "Compare GaussMoments analytic moment values to numerical integrations.\n"
	"Usage: testGaussMoments <flux> [sigma=1.5] [e=0.] [sigmaW=1.] [beta=0.] [x0=0.] [y0=0.]\n"
	" where you give values but will take noted default values starting at rhs if not all args are\n"
	" given.\n"
	" Output is comparison of GaussMoments moment values vs numerical integration.\n"
	" Discrepancies will be marked with ""FAILURE"" \n"
	" You have given no arguments so a standard non-default set is being used." << endl;
    } 
    if (argc > 1) flux  = atof(argv[1]);
    if (argc > 2) sigma = atof(argv[2]);
    if (argc > 3) e     = atof(argv[3]);
    if (argc > 4) sigmaW= atof(argv[4]);
    if (argc > 5) beta  = atof(argv[5]) * 3.14159 / 180.;
    if (argc > 6) x0    = atof(argv[6]);
    if (argc > 7) y0    = atof(argv[7]);

    const double TOLERANCE = 1e-10;   // Absolute tolerance
    const double FTOL = 1e-4;         // Relative tolerance (both must pass)

    // Do the numerical integration:
    double dk = 0.01;
    double kmax = 10.;

    // Note that for GaussMoments, 2*sigma^2 is the trace of the covariance matrix of the Gaussian.
    double sigxx = sigma*sigma*(1+e*cos(2*beta)) + sigmaW*sigmaW;
    double sigyy = sigma*sigma*(1-e*cos(2*beta)) + sigmaW*sigmaW;
    double sigxy = sigma*sigma*e*sin(2*beta);
    CMatrix sum(maxN+1, maxN+1, 0.);
    for (double kx = -kmax; kx<=kmax; kx+=dk) {
      for (double ky = -kmax; ky<=kmax; ky+=dk) {
	DComplex z = exp( DComplex(-0.5*(sigxx*kx*kx + sigyy*ky*ky + 2*sigxy*kx*ky),
				   -(kx * x0 + ky*y0)));
	sum += z*kpow(DComplex(kx,ky));
      }
    }

    sum *= flux * dk * dk;

    // Compare to analytic class results
    GaussMoments gm(flux, sigma, e, sigmaW, beta, x0, y0);

    bool failure = false;
    cout << "p,q: analytic --- numerical  ===== diff" << endl;
    for (int p=0; p<=maxN; ++p)
      for (int q=0; q<=p && p+q<=maxN; q++) {
	DComplex analytic = gm.moment(p,q);
	DComplex numerical = sum(p,q);
	DComplex diff = analytic - numerical;
	if (abs(real(diff)) > max(TOLERANCE, FTOL*abs(real(analytic))) ||
	    abs(imag(diff)) > max(TOLERANCE, FTOL*abs(imag(analytic)))) {
	  cout << "FAILURE:" << endl;
	  failure = true;
	}
	cout << p << "," << q << ": " << analytic << " -- " << numerical << " == " << diff << endl;
      }
    exit(failure ? 1 : 0);
  } catch (std::runtime_error &m) {
    quit(m,1);
  }
}
