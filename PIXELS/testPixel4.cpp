// Compare actual covariance matrix of moments to the calculated one,
// for both direct-to-k-space PixelGalaxy and one from a drawn image.
// The latter includes convolution and removal of a PSF.
#include "Std.h"
#include "Moments.h"
#include "KWeight.h"
#include "Galaxy.h"
#include "PixelGalaxy.h"
#include "Image.h"
#include "SBProfile.h"
#include "SBPixel.h"
#include "SbPsf.h"
#include "Random.h"

using namespace bfd;
using namespace sbp;

typedef MomentIndices<> MI;

const int UseMoments = USE_ALL_MOMENTS;

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
      sigmaW = 2.;
      beta = 40. * 3.1415/180.;
      x0 = 0.1;
      y0 = -1.2;
      psffwhm = 2.;
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

    long ncov = 10000;

    double noise = 1.;
    flux = 0.;
    ran::UniformDeviate ud;
    //**/GaussianWeight kw(sigmaW);
    /**/KSigmaWeight kw(sigmaW/sqrt(8.),4);

    // Make a Moffat PSF
    SBMoffat psf0(3.5, 6.);
    psf0.setFWHM(psffwhm);
    //SBProfile* psf = psf0.shear(0.2, -0.1);
    SBProfile* psf = psf0.shear(0., -0.);
    SbPsf psf1(*psf);
    cerr << "psf1 kmax: " << psf1.kMax() << endl;

    SBGaussian src(flux, sigma);
    Shear s;
    s.setEBeta(e,beta);
    Ellipse distortion(s, 0., Position<double>(x0,y0));
    SBProfile* lensed = src.distort(distortion);

    // Make our direct k-space version of the galaxy - no noise
    PixelGalaxy<UseMoments>* pg1 = SbGalaxy<UseMoments>(kw, *lensed, psf1, ud, noise, false);
    cout << "----SbGalaxy predicted covariance:" << endl;
    cout << pg1->getCov() << endl;
    cout << pg1->getMoments() << endl;
    delete pg1;

    DVector msum(6,0.);
    DMatrix covsum(6,6,0.);
    DVector mm(6);
    for (long i=0; i<ncov; i++) {
      PixelGalaxy<UseMoments>* pg1 = SbGalaxy<UseMoments>(kw, *lensed, psf1, ud, noise, true);
      MI::MVector m = pg1->getMoments();
      for (int i=0; i<6; i++) mm[i]=m[i];
      delete pg1;
      msum += mm;
      covsum += mm^mm;
    }
    msum /= (double) ncov;
    covsum /= (double) ncov;
    covsum -= msum ^ msum;
    cout << "-----SBGalaxy empirical covariance:" << endl;
    cout << covsum << endl;
    
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
    {
      PixelGalaxy<> pg2(kw, data, map, psf1, noise * (dx*dx));
      cout << "----PixelGalaxy predicted covariance:" << endl;
      cout << pg2.getCov() << endl;
      cout << pg2.getMoments() << endl;
    }
    ran::GaussianDeviate gd(ud);
    double nn = sqrt(noise)*dx;
    msum.setZero();
    covsum.setZero();
    for (long i=0; i<ncov; i++) {
      img::Image<> noisy = data.duplicate();
      for (int i=noisy.yMin(); i<=noisy.yMax(); i++)
	for (int j=noisy.xMin(); j<=noisy.xMax(); j++)
	  noisy(j,i) += nn*gd;
      PixelGalaxy<UseMoments> pg2 = PixelGalaxy<UseMoments>(kw, noisy, map, psf1, noise*(dx*dx));
      MI::MVector m = pg2.getMoments();
      for (int i=0; i<6; i++) mm[i]=m[i];
      msum += mm;
      covsum += mm^mm;
    }
    msum /= (double) ncov;
    covsum /= (double) ncov;
    covsum -= msum ^ msum;
    cout << "-----PixelGalaxy empirical covariance:" << endl;
    cout << covsum << endl;

  } catch (std::runtime_error& m) {
    quit(m,1);
  }
}
