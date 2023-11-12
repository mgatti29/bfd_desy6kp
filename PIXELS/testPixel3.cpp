// Code that compares moments taken directly into k space with SbGalaxy to
// those from a galaxy drawn with SBProfile and measured as a PixelGalaxy.
// The latter includes convolution and removal of a PSF.
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
#include "Random.h"

using namespace bfd;
using namespace sbp;

typedef MomentIndices<> MI;

const int UseMoments = USE_ALL_MOMENTS;

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
	<< "Compare GaussianGalaxy analytic moment values to numerical values from galaxy drawn\n"
	" with SBProfile and analyzed as a PixelGalaxy; Moffat PSF used.\n"
	" First time uses the direct k-space evaluation of the PSF.\n"
	" Second time draws the PSF, then makes PSF from SBPixel.\n"
	" Third time tells PixelGalaxy that origin is at ctr of galaxy.\n"
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
    ran::UniformDeviate ud;
    //**/GaussianWeight kw(sigmaW);
    /**/KSigmaWeight kw(sigmaW/sqrt(8.),4);

    // Make a Moffat PSF
    SBMoffat psf0(3.5, 6.);
    psf0.setFWHM(psffwhm);
    SBProfile* psf = psf0.shear(0.2, -0.1);
    SbPsf psf1(*psf);
    cerr << "psf1 kmax: " << psf1.kMax() << endl;

    SBGaussian src(flux, sigma);
    Shear s;
    s.setEBeta(e,beta);
    Ellipse distortion(s, 0., Position<double>(x0,y0));
    SBProfile* lensed = src.distort(distortion);

    // Make our direct k-space version of the galaxy - no noise
    PixelGalaxy<UseMoments>* pg1 = SbGalaxy<UseMoments>(kw, *lensed, psf1, ud, 0.);
    
    // Now draw the image and read it back as PixelGalaxy
    SBConvolve observed(*lensed, *psf);
    double dx=0;
    img::Image<> data = observed.draw();
    data.getHdrValue("DX",dx);
    data *= (dx*dx);	// Turn into SB in flux per pixel.
    cerr << "Image dimensions: " << data.getBounds() << endl;
    cerr << "Image dx: " << dx << endl;

    DVector2 origin(0.);
    Affine map(dx,origin);
    PixelGalaxy<> pg2(kw, data, map, psf1, noise * (dx*dx));

    bool failure = false;
    if (false) {
      Moments<> numeric = pg1->getMoments();
      Moments<> analytic = pg2.getMoments();
      cout << "Moments: " << endl;
      failure = compare(analytic, numeric) || failure;

      /**/ cout << "Covariance analytic: " << pg1->getCov() << endl;
      cout << "Covariance numerical: " << pg2.getCov() << endl;
      /**/
    }

    MomentDerivs<> numeric = pg2.getDerivs();
    MomentDerivs<> analytic = pg1->getDerivs();

    cout << "---- Using analytic PSF---" << endl;

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

    // Do it again, this time with PSF given as an image
    img::Image<> psfIm = psf->draw();
    fft::Lanczos interp1(5,true);
    fft::InterpolantXY xInterp(interp1);
    cerr << "Interpolant urange: " << xInterp.urange() << endl;
    psfIm.getHdrValue("DX",dx);
    cerr << "Image pixel scale: " << dx << " " << endl;

    SBPixel pixPsf(psfIm, xInterp);
    cerr << "kMax of pixPsf: " << pixPsf.maxK() << endl;
    SbPsf psf2(pixPsf);
    cerr << "psf2 kmax: " << psf2.kMax() << endl;

    PixelGalaxy<> pg3(kw, data, map, psf2, noise * (dx*dx));

    cout << "---- Now using pixelated PSF---" << endl;
    numeric = pg3.getDerivs();

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


    cout << "---- Now shift move origin to center---" << endl;
    // Comparing a new direct k-space calc of centered galaxy with shift() of the original
    distortion = Ellipse(s, 0., Position<double>(0.,0.));
    delete lensed;
    lensed = src.distort(distortion);
    PixelGalaxy<UseMoments>* pg4 = SbGalaxy<UseMoments>(kw, *lensed, psf1, ud, 0.);

    TemplateGalaxy<>* pg5 = dynamic_cast<TemplateGalaxy<>*> (pg1->getShifted(-x0, -y0));
    numeric = pg5->getDerivs();
    analytic = pg4->getDerivs();

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




    data.shift(1,1);
    img::FitsImage<>::writeToFITS("test.fits",data);

    exit(failure ? 1 : 0);

  } catch (std::runtime_error& m) {
    quit(m,1);
  }
}
