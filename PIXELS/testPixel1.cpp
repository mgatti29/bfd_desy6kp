// Code that compares the analytic GaussianGalaxy moments/derivs to numerical values
// obtained from a galaxy drawn with SBProfile and measured as a PixelGalaxy
#include "Std.h"
#include "Moments.h"
#include "KWeight.h"
#include "Galaxy.h"
#include "PixelGalaxy.h"
#include "Image.h"
#include "SBProfile.h"
#include "FitsImage.h"

using namespace bfd;
using namespace sbp;

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
	<< "Compare GaussianGalaxy analytic moment values to numerical values from galaxy drawn\n"
	" with SBProfile and analyzed as a PixelGalaxy; Delta-function PSF.\n"
	"Usage: testPixel1 <flux> [sigma=1.5] [e=0.] [sigmaW=1.] [beta=0.] [x0=0.] [y0=0.]\n"
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
    GaussianGalaxy<> gal(kw, flux, sigma*pow(1-e*e,-0.25), e, beta, x0, y0, noise);

    // Use delta-function PSF now.
    DeltaPsf psf;

    SBGaussian src(flux, sigma);
    Shear s;
    s.setEBeta(e,beta);
    Ellipse distortion(s, 0., Position<double>(x0,y0));
    SBProfile* lensed = src.distort(distortion);

    //**    double dx = 0.2;
    double dx = 1.;
    img::Image<> data = lensed->draw(dx);
    data *= (dx*dx);	// Turn into SB in flux per pixel.
    cerr << "Image dimensions: " << data.getBounds() << endl;
    cerr << "Origin value: " << data(0,0) << endl;

    DMatrix22 a(0.);
    a(0,0) = a(1,1) = dx;
    DVector2 origin(0.);
    Affine map(a,origin);

    PixelGalaxy<> pg(kw, data, map, psf, noise * (dx*dx));

    data.shift(1,1);
    img::FitsImage<>::writeToFITS("test.fits",data);

    bool failure = false;
    {
      Moments<> numeric = pg.getMoments();
      Moments<> analytic = gal.getMoments();
      cout << "Moments: " << endl;
      failure = compare(analytic, numeric) || failure;

      /**/ cout << "Covariance analytic: " << gal.getCov() << endl;
      cout << "Covariance numerical: " << pg.getCov() << endl;
      /**/
    }

    MomentDerivs<> numeric = pg.getDerivs();
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
