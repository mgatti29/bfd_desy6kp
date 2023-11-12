// First test of Great3 interfaces.
#include "Great3.h"
#include "Interpolant.h"
#include "FitsImage.h"
#include "PixelGalaxy.h"

using namespace bfd;

int main(int argc,
	 char *argv[])
{
  string psffile = argv[1];
  string galfile = argv[2];
  int nx = argc>3 ? atoi(argv[3]) : 0;
  int ny = argc>4 ? atoi(argv[4]) : 0;
  int stampSize = 48;

  typedef MomentIndices<> MI;
  try {
    fft::SincInterpolant interp1d;
    fft::InterpolantXY interp2d(interp1d);
    double hlr;
    SbPsf psf(Great3Psf(psffile, interp2d, 0, 0, stampSize,hlr));

    double kmax = psf.kMax();
    for (double k=0.; k<kmax; k+=kmax/100.)
      cout << k 
	   << " " << abs(psf.kValue(k,0.)) 
	   << " " << psf.kValue(k,0.)
	   << endl;

    // Make a weight function
    KSigmaWeight kw(1./1.5, 4);  // Cutoff at k=1.5, (1-k^2/kcut^2)^4
    double noise = 0.00034;

    img::FitsImage<> fi(galfile, FITS::ReadOnly, 0);
    Bounds<int> b(nx*stampSize+1,(nx+1)*stampSize,ny*stampSize+1,(ny+1)*stampSize);
    img::Image<> stamp = fi.extract(b);

    PixelGalaxy<> pg = Great3Galaxy<USE_ALL_MOMENTS>(stamp, psf, kw, noise);

    cout << "Moments: " << pg.getMoments() << endl;

    cout << "dMdx: " << pg.dMdx() << endl;

    cout << "DG1: " << pg.getDerivs().col(Pqr::DG1) << endl;
    
    cout << "Covariance matrix: " << pg.getCov() << endl;
    cout << "sig Mx, My: " << sqrt(pg.getCov()(MI::CX,MI::CX))
	 << " " << sqrt(pg.getCov()(MI::CY,MI::CY))
	 << endl;
      

    // Try a Newton iteration on the centroid
    DVector2 dm;
    dm[0] = pg.getMoments()[MI::CX];
    dm[1] = pg.getMoments()[MI::CY];
    DMatrix22 dmdx;
    dmdx(0,0) = pg.dMdx()[MI::CX];
    dmdx(0,1) = pg.dMdy()[MI::CX];
    dmdx(1,0) = pg.dMdx()[MI::CY];
    dmdx(1,1) = pg.dMdy()[MI::CY];
    dm /= dmdx;
    /**/cerr << "dm: " << dm << endl;
    Galaxy<>* shifted = pg.getShifted(-dm[0],-dm[1]);
    /**/cerr << "gets moments " << shifted->getMoments() << endl;

  } catch (std::runtime_error& e) {
    quit(e,1);
  }
}


