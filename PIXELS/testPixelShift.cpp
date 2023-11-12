// Code that compares the analytic GaussianGalaxy translation/derivatives
// to values derived from a drawn PixelGalaxy
#include "Std.h"
#include "Moments.h"
#include "KWeight.h"
#include "Galaxy.h"
#include "PixelGalaxy.h"
#include "Image.h"
#include "SBProfile.h"
#include "FitsImage.h"
#include "SBPixel.h"
#include "SbPsf.h"

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
    double psffwhm=1.3;

    if (argc<2 || argc > 9) {
      flux = 2.;
      sigma = 3.;
      e = 0.7;
      sigmaW = 0.5;
      beta = 40. * 3.1415/180.;
      x0 = 0.1;
      y0 = -1.2;
      psffwhm = 1.3;
      cerr 
	<< "Compare GaussianGalaxy analytic position derivatives and finite shifts to numerical ones drawn\n"
	" with SBProfile and analyzed as a PixelGalaxy; Moffat PSF used.\n"
	" PSF is drawn, then makes PSF from SBPixel.\n"
	"Usage: testPixel2 <flux> [sigma=1.5] [e=0.] [sigmaW=1.] [beta=0.] \n"
	"                  [x0=0.] [y0=0.] [psffwhm=1.3]\n"
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
    if (argc > 8) psffwhm = atof(argv[8]);

    double noise = 1.;
    GaussianWeight kw(sigmaW);
    GaussianGalaxy<> gal(kw, flux, sigma*pow(1-e*e,-0.25), e, beta, x0, y0, noise);

    // Make a Moffat PSF
    SBMoffat psf0(3.5, 6.);
    psf0.setFWHM(psffwhm);
    SBProfile* psf = psf0.shear(0.2, -0.1);

    SBGaussian src(flux, sigma);
    Shear s;
    s.setEBeta(e,beta);
    Ellipse distortion(s, 0., Position<double>(x0,y0));
    SBProfile* lensed = src.distort(distortion);
    SBConvolve observed(*lensed, *psf);

    double dx=0;
    img::Image<> data = observed.draw();
    data.getHdrValue("DX",dx);
    data *= (dx*dx);	// Turn into SB in flux per pixel.
    cerr << "Image dimensions: " << data.getBounds() << endl;
    cerr << "Image dx: " << dx << endl;

    DVector2 origin(0.);
    Affine map(dx,origin);

    bool failure = false;

    // Do it again, this time with PSF given as an image
    img::Image<> psfIm = psf->draw();
    fft::Lanczos interp1(5,true);
    fft::InterpolantXY xInterp(interp1);
    cerr << "Interpolant urange: " << xInterp.urange() << endl;
    psfIm.getHdrValue("DX",dx);
    cerr << "Image pixel scale: " << dx << " " << endl;

    SBPixel pixPsf(psfIm, xInterp);
    cerr << "kMax of pixPsf: " << pixPsf.maxK() << endl;
    SbPsf psf1(pixPsf);
    cerr << "psf1 kmax: " << psf1.kMax() << endl;

    PixelGalaxy<> pg1(kw, data, map, psf1, noise * (dx*dx));

    cout << "---- Moments and position derivatives pixelated PSF---" << endl;

    cout << "Moments: " << endl;
    failure = compare(gal.getMoments(), pg1.getMoments()) || failure;

    cout << "x derivs: " << endl;
    failure = compare(gal.dMdx(), pg1.dMdx()) || failure;
    cout << "y derivs: " << endl;
    failure = compare(gal.dMdy(), pg1.dMdy()) || failure;

    cout << "Finite centroid shift moments: " << endl;
    double shiftx = -0.1;
    double shifty = +0.13;
    Galaxy<>* gal2 = gal.getShifted(shiftx,shifty);
    Galaxy<>* pg2 = pg1.getShifted(shiftx,shifty);
    failure = compare(gal2->getMoments(), pg2->getMoments()) || failure;

    delete gal2;
    delete pg2;

    data.shift(1,1);
    img::FitsImage<>::writeToFITS("test.fits",data);

    exit(failure ? 1 : 0);

  } catch (std::runtime_error& m) {
    quit(m,1);
  }
}
