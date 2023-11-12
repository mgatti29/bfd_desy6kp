// Some tests of the SbPsf wrapper
#include "Std.h"
#include "SbPsf.h"
#include "Interpolant.h"
#include "SBPixel.h"
using namespace bfd;
using namespace sbp;

int main(int argc,
	 char *argv[])
{
  try {
    SBAiry psf(1.,0.3);
    double threshold = argc>1 ? atof(argv[1]) : 1e-4;
    cerr << "Initial kMax: " << psf.maxK() << endl;
    cerr << "Input dk: " << psf.stepK() << endl;
    SbPsf s(psf, threshold);
    cerr << "SbPsf kmax: " << s.kMax() << endl;

    img::Image<> ipsf = psf.draw();
    //    fft::Cubic interp1;
    //    fft::Lanczos interp1(3,true);
    fft::SincInterpolant interp1;
    fft::InterpolantXY xInterp(interp1);
    cerr << "Interpolant urange: " << xInterp.urange() << endl;
    double dx;
    ipsf.getHdrValue("DX",dx);
    cerr << "Image pixel scale: " << dx << " " << 2*PI *xInterp.urange() / dx << endl;

    SBPixel pixPsf(ipsf, xInterp);
    cerr << "kMax of pixPsf: " << pixPsf.maxK() << endl;
    SbPsf s2(pixPsf, threshold);
    cerr << "S2 kmax: " << s2.kMax() << endl;

    double kx = 0.3 * s2.kMax();
    double ky = -0.6*kx;
    cerr << "kValue from original: " << psf.kValue(Position<double>(kx,ky)) << endl;
    cerr << "kValue from s2: " << s2.kValue(kx,ky) << endl;

  } catch (std::runtime_error& m) {
    quit(m,1);
  }
}
