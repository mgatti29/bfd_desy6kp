// A minimal driver program to calculate BFD moments of a pixellated galaxy with
// a pixelated PSF.

#include <fstream>

#include "Great3.h"
#include "Interpolant.h"
#include "PixelGalaxy.h"
#include "StringStuff.h"
#include "Pset.h"
#include "Image.h"
#include "Table.h"
#include "FitsTable.h"
#include "FitsImage.h"
#include "SBPixel.h"

string usage = "basicMeasure: Measure moments of galaxies in an image with fixed PSF\n"
  "       and save these to a FITS binary table.\n"
  "usage: basicMeasure [parameter file] [-parameter [=] value...]\n"
  "       Parameter files read in order of command line, then command-line parameters,\n"
  "       later values override earlier.\n"
  "stdin: none\n"
  "     Run with -help to get parameter list\n"
  "stdout: Diagnostic information";

using namespace bfd;

int main(int argc,
	 char *argv[])
{
  // These two lines inform all the code/templates that we'll be using all 6 moments
  const int UseMoments=USE_ALL_MOMENTS;
  typedef MomentIndices<UseMoments> MI;

  // Here we declare all the variables that will hold values of the program's
  // running parameters
  string targetFile;		// The FITS file holding galaxy images
  int targetStampSize;		// Pixels on each side of each postage stamp
  double targetNoise;		// The RMS noise added to each pixel

  string psfFile;		// The FITS file holding PSF image

  double weightKSigma;		// These two parameters describe the k-space weight
  int weightN;

  string outfile;		// Name of the output file holding FITS table of moments

  /***** The following uses my code to set up parameters to be read from
   ****  parameter files or from command-line options
   ****/
  Pset parameters;
  {
      const int def=PsetMember::hasDefault;
      const int low=PsetMember::hasLowerBound;
      const int up=PsetMember::hasUpperBound;
      const int lowopen = low | PsetMember::openLowerBound;
      const int upopen = up | PsetMember::openUpperBound;

      parameters.addMember("targetFile",&targetFile, def,
			   "Target image filename", 
			   "target.fits");
      parameters.addMember("targetStampSize",&targetStampSize, def | low,
			   "Pixels in each target postage stamp", 48, 16);
      parameters.addMember("targetNoise", &targetNoise, def | lowopen,
			   "Pixel variance in target images", 0.00496, 0.);

      parameters.addMember("weightKSigma", &weightKSigma, def | lowopen,
			   "Length scale in the k-space weight", 1.35, 0.);
      parameters.addMember("weightN", &weightN, def | low,
			   "Power-law index of the k-space weight", 4, 2);

      parameters.addMember("psfFile",&psfFile, def,
			   "FITS file holding the PSF for the targets",
			   "psf.fits");

      parameters.addMember("outfile",&outfile, def,
			   "FITS file holding the measured moments",
			   "moments.fits");
  }

  parameters.setDefault();
  
  try {
    int positionalArguments;
    try {
      // First read the command-line arguments so we know how many positional
      // arguments precede them.
      positionalArguments = parameters.setFromArguments(argc, argv);
    } catch (std::runtime_error &m) {
      // An error here might indicate someone entered "-help" or something
      cerr << usage << endl;
      cerr << "#---- Parameter defaults: ----" << endl;
      parameters.dump(cerr);
      quit(m,1);
    }
    for (int i=1; i<positionalArguments; i++) {
      // Open & read all specified input files
      ifstream ifs(argv[i]);
      if (!ifs) {
	cerr << "Can't open parameter file " << argv[i] << endl;
	cerr << usage << endl;
	exit(1);
      }
      try {
	parameters.setStream(ifs);
      } catch (std::runtime_error &m) {
	cerr << "In file " << argv[i] << ":" << endl;
	quit(m,1);
      }
    }
    // And now re-read the command-line arguments so they take precedence
    parameters.setFromArguments(argc, argv);

    /*** Now have read in all the parameters for this program.  The next
     *** two lines will just print out the selected parameters to 
     *** standard output.
     ***/
    cout << "#" << stringstuff::taggedCommandLine(argc,argv) << endl;
    parameters.dump(cout);

    // Our PSF will arrive as a pixelated (sampled) image.  We need to define 
    // what values it takes between the samples in the image, in order to fully 
    // specify our PSF.  These lines define the interpolant between PSF samples to
    // be a 2d sinc function, i.e. we are going to assume that the PSF image
    // is Nyquist-sampled.
    fft::SincInterpolant interp1d;
    fft::InterpolantXY interp2d(interp1d);

    // Now we're going to turn the PSF image into an SBProfile
    sbp::SBProfile *psfProfile=0;	// Make a null pointer to it
    {
      // First read the image holding PSF
      img::FitsImage<> fi(psfFile, FITS::ReadOnly, 0);
      const img::Image<> data = fi.use();

      // Determine its centroid using a Gaussian weight near the center
      double x0 = 0.5*(data.xMin()+data.xMax());	// 1st guess at center
      double y0 = 0.5*(data.yMin()+data.yMax());
      double sigma = (data.xMax()-data.xMin()+1) / 12.;
      double halfLightRadius;
      centroid(data, sigma, x0, y0, halfLightRadius);

      // Make an SBProfile from the PSF image
      sbp::SBPixel pixPsf(data, interp2d);
      // Normalize to unit flux
      pixPsf.setFlux(1.);

      // The pixel value where the FFT is placing its origin is 1 past halfway:
      int xOriginFT = (data.xMin() + data.xMax() + 1)/2;
      int yOriginFT = (data.yMin() + data.yMax() + 1)/2;

      // Our output SBProfile just shifts the phases to put origin at the centroid
      psfProfile = pixPsf.shift(xOriginFT-x0, yOriginFT-y0);

      // We got what we want out of the PSF image.  It will be destroyed when
      // we close this block.
    }

    // Now we make an SbPsf from the SBProfile
    SbPsf targetPsf(*psfProfile);
    // That made a copy of the psfProfile, so we can delete our old one.
    delete psfProfile;
    psfProfile=0;

    // Make the weight function
    KSigmaWeight kw(weightKSigma, weightN);

    // Covariances are all the same in this code since PSF, noise, and weight are
    // fixed.  So we'll make one covariance matrix from the first stamp, and save it.
    MomentCovariance<UseMoments> cov;
    bool firstGalaxy = true;

    // These vectors will hold the x,y positions for each postage stamp's galaxy,
    // and the moments.
    vector<Moments<UseMoments> > mvec;
    vector<double> vx;
    vector<double> vy;

    // Open the FITS file containing our target galaxies
    img::FitsImage<> fi(targetFile, FITS::ReadOnly, 0);

    // Loop through all postage stamps in the file, collecting their
    // moments.  
    for (int iy0 = 1; iy0<fi.getBounds().getYMax(); iy0+=targetStampSize) {
      for (int ix0 = 1; ix0<fi.getBounds().getXMax(); ix0+=targetStampSize) {
	Bounds<int> b(ix0,ix0+targetStampSize-1,
		      iy0,iy0+targetStampSize-1);
	// First read out the pixels for this stamp
	const img::Image<> stamp = fi.use(b);

	// Read the galaxy from the pixels, and return its centroid too.
	double cx, cy;
	PixelGalaxy<UseMoments> pg = Great3Galaxy<UseMoments>(stamp, 
							      targetPsf, 
							      kw, 
							      targetNoise,
							      cx,cy);

	// If this is the first galaxy, we'll ask for the moment covariance matrix
	if (firstGalaxy) {
	  cov = pg.getCov();
	  firstGalaxy = false;
	}

	// Now we're going to adjust the coordinate origin so that the 
	// X and Y moments are nulled.
	double dx, dy;
	  
	TemplateGalaxy<>* shifted = newtonShift(pg, dx, dy, 2);

	// Now save the center and moments of this galaxy on our list
	mvec.push_back(shifted->getMoments());
	vx.push_back(cx + dx);
	vy.push_back(cy + dy);

	// And clean up the shifted galaxy that we made:
	delete shifted;
      } // end ix loop
    } // iy loop

    // Now produce the output FITS file.
    // Primary extension is a small image giving the covariance matrix
    {
      img::Image<float> covimg(MI::N, MI::N);
      for (int i=0; i<MI::N; i++)
	for (int j=0; j<MI::N; j++)
	  covimg(i+1,j+1) = cov(i,j);
      covimg.shift(1,1);
      img::FitsImage<float> fi(outfile, 
			       FITS::ReadWrite + FITS::Create + FITS::OverwriteFile,
			       0);
      fi.copy(covimg);
    }
    
    // Next make a table holding positions and moments for each galaxy
    long nRows = mvec.size();

    // Create FTable
    img::FTable tab;

    // Make Columns for:
    // x, y
    tab.addColumn(vx, "X");
    tab.addColumn(vy, "Y");

    // moments
    {
      vector<float> vm(MI::N);
      vector<vector<float> > vvf;
      for (long i=0; i<nRows; i++) {
	for (int j=0; j<MI::N; j++)
	  vm[j] = mvec[i][j];
	vvf.push_back(vm);
      }
      tab.addColumn(vvf, "MOMENTS", MI::N);
    }

    // Write to FITS
    FITS::FitsTable ft(outfile, FITS::ReadWrite + FITS::Create, 1);
    ft.copy(tab);

  } catch (std::runtime_error& e) {
    quit(e,1);
  }
}


