// Code that compares the analytic GaussianGalaxy moments/derivs to a numerical integration 
// with the WKMoments class
#include "Std.h"
#include "Moments.h"
#include "KWeight.h"
#include "Galaxy.h"
#include "PixelGalaxy.h"

using namespace bfd;

typedef MomentIndices<> MI;

bool
compare(const DVector& analytic, const DVector& numeric) {
  const double TOLERANCE = 1e-4;
  const double FTOL = 1e-3;
  int N = analytic.size();
  bool failure = false;
  for (int i=0; i<N; i++) {
    double diff = analytic[i] - numeric[i];
    if (abs(diff) > max(TOLERANCE, FTOL*abs(analytic[i]))) {
      failure = true;
    }
  }
  if (failure) cout << "FAILURE:" << endl;
  cout << "Analytic: ";
  for (int i=0; i<N; i++) cout << fixed << setprecision(6) << analytic[i] << " ";
  cout << endl;
  cout << "Numeric:  ";
  for (int i=0; i<N; i++) cout << fixed << setprecision(6) << numeric[i] << " ";
  cout << endl;
  return failure;
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
      flux = 2.;
      sigma = 3.;
      e = 0.7;
      sigmaW = 0.5;
      beta = 40. * 3.1415/180.;
      x0 = 0.1;
      y0 = -1.2;
      cerr 
	<< "Compare GaussianGalaxy analytic moment values to numerical values from WK classes.\n"
	"Usage: testWK <flux> [sigma=1.5] [e=0.] [sigmaW=1.] [beta=0.] [x0=0.] [y0=0.]\n"
	" where you give values but will take noted default values starting at rhs if not all args are\n"
	" given.\n"
	" Output is comparison of analytic moment values vs numerical integration.\n"
	" Discrepancies will be marked with ""FAILURE"" and program will return nonzero value.\n"
	" You have given no arguments so a standard non-default set is being used." << endl;
    } 
    if (argc > 1) flux  = atof(argv[1]);
    if (argc > 2) sigma = atof(argv[2]);
    if (argc > 3) e     = atof(argv[3]);
    if (argc > 4) sigmaW= atof(argv[4]);
    if (argc > 5) beta  = atof(argv[5]) * 3.14159 / 180.;
    if (argc > 6) x0    = atof(argv[6]);
    if (argc > 7) y0    = atof(argv[7]);

    double noise = 1.;
    GaussianWeight kw(sigmaW);
    GaussianGalaxy<> gal(kw, flux, sigma, e, beta, x0, y0, noise);

    double dk = 0.01;
    double kmax = kw.kMax();

    double sigxx = sigma*sigma*(1+e*cos(2*beta));
    double sigyy = sigma*sigma*(1-e*cos(2*beta));
    double sigxy = sigma*sigma*e*sin(2*beta);

    vector<double> vkx;
    vector<double> vky;
    vector<double> vre;
    vector<double> vim;

    // DC term first:
    vkx.push_back(0.);
    vky.push_back(0.);
    vre.push_back(flux);
    vim.push_back(0.);

    for (double kx = 0; kx<=kmax; kx+=dk) 
      for (double ky = -kmax; ky<=kmax; ky+=dk) {
	if ( (kx==0. && ky<0.) || (kx*kx+ky*ky) > kmax*kmax ) continue;
	DComplex z = exp( DComplex(-0.5*(sigxx*kx*kx + sigyy*ky*ky + 2*sigxy*kx*ky),
				   -(kx * x0 + ky*y0)));
	z *= flux;
	vkx.push_back(kx);
	vky.push_back(ky);
	vre.push_back(z.real());
	vim.push_back(z.imag());
      }

    KData kd(vkx.size());
    kd.d2k = dk*dk;
    kd.dcIndex = 0;
    for (int i = 0; i<vkx.size(); i++) {
      kd.kx[i] = vkx[i];
      kd.ky[i] = vky[i];
      kd.re[i] = vre[i];
      kd.im[i] = vim[i];
    }

    bool failure = false;
    {
      Moments<> numeric = WKMoments<>(kw, kd).getMoments(kd);
      Moments<> analytic = gal.getMoments();
      cout << "Moments: " << endl;
      failure = compare(analytic, numeric) || failure;

      /** cout << "Covariance analytic: " << gal.getCov() << endl;
      KData psf(kd);
      psf.re.setAllTo(1.);
      psf.im.setZero();
      cout << "Covariance numerical: " << WKMoments<>(kw,kd).getCov(psf) << endl;
      /**/
    }

    MomentDerivs<> numeric = WKDerivs<>(kw,kd).getDecomposition(kd).getDerivs();
    MomentDerivs<> analytic = gal.getDerivs();

    cout << "Moments via decomp: " << endl;
    failure = compare(analytic.col(Pqr::P), numeric.col(Pqr::P)) || failure;

    cout << "G1 derivs: " << endl;
    failure = compare(analytic.col(Pqr::DG1), numeric.col(Pqr::DG1)) || failure;
    cout << "G2 derivs: " << endl;
    failure = compare(analytic.col(Pqr::DG2), numeric.col(Pqr::DG2)) || failure;

    cout << "G1G1 derivs: " << endl;
    failure = compare(analytic.col(Pqr::D2G1G1), numeric.col(Pqr::D2G1G1)) || failure;
    cout << "G2G2 derivs: " << endl;
    failure = compare(analytic.col(Pqr::D2G2G2), numeric.col(Pqr::D2G2G2)) || failure;
    cout << "G1G2 derivs: " << endl;
    failure = compare(analytic.col(Pqr::D2G1G2), numeric.col(Pqr::D2G1G2)) || failure;

    exit(failure ? 1 : 0);

  } catch (std::runtime_error& m) {
    quit(m,1);
  }
}
