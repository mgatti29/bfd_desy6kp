// Program to estimate a shear S/N level for a given choice of weight by
// looking at the shear derivatives of the template set, relative to noise level.
#include <fstream>

#include "Great3.h"
#include "Interpolant.h"
#include "FitsImage.h"
#include "PixelGalaxy.h"
#include "StringStuff.h"
#include "Pset.h"
#include "Statistics.h"

string usage = "greatPsfSize: Find PSF size on a GREAT3 psf image\n"
  "usage: greatPsfSize <fitsname>\n"
  "stdout: PSF half-light radius";

using namespace bfd;

int main(int argc,
	 char *argv[])
{
  const int UseMoments=USE_ALL_MOMENTS;
  typedef MomentIndices<UseMoments> MI;

  try {
    string psfFile = argv[1];
    // Acquire the PSF;
    fft::SincInterpolant interp1d;
    fft::InterpolantXY interp2d(interp1d);

    double hlr;
    double hlr2;
    int psfStampSize = 48;
    Great3Psf(psfFile, interp2d, 0, 0, psfStampSize, hlr);
    Great3Psf(psfFile, interp2d, 2, 2, psfStampSize, hlr2);
    cout << psfFile
	 << " " << hlr
	 << " " << hlr2
	 << endl;
  } catch (std::runtime_error& e) {
    quit(e,1);
  }
}


