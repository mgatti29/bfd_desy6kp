// Program to measure moments of galaxies in a GREAT3 image with fixed PSF.
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

string usage = "greatMeasure: Measure moments of galaxies in a GREAT3 image with fixed PSF\n"
  "       and save these to a FITS binary table.\n"
  "usage: greatMeasure [parameter file] [-parameter [=] value...]\n"
  "       Parameter files read in order of command line, then command-line parameters,\n"
  "       later values override earlier.\n"
  "stdin: none\n"
  "     Run with -help to get parameter list\n"
  "stdout: Diagnostic information";

using namespace bfd;

int main(int argc,
	 char *argv[])
{
  const int UseMoments=USE_ALL_MOMENTS;
  typedef MomentIndices<UseMoments> MI;

  string targetFiles;
  int targetStampSize;
  double targetNoise;

  string targetPsfFile;
  int targetPsfNx;
  int targetPsfNy;
  int targetPsfStampSize;

  double weightKSigma;
  int weightN;

  string outfile;

  Pset parameters;
  {
      const int def=PsetMember::hasDefault;
      const int low=PsetMember::hasLowerBound;
      const int up=PsetMember::hasUpperBound;
      const int lowopen = low | PsetMember::openLowerBound;
      const int upopen = up | PsetMember::openUpperBound;

      parameters.addMemberNoValue("TARGETS:",0,
				  "The target images");
      parameters.addMember("targetFiles",&targetFiles, def,
			   "String(s) to glob to get target image filenames", 
			   "image*fits");
      parameters.addMember("targetStampSize",&targetStampSize, def | low,
			   "Pixels in each target postage stamp", 48, 16);
      parameters.addMember("targetNoise", &targetNoise, def | lowopen,
			   "Pixel variance in target images", 0.00496, 0.);

      parameters.addMemberNoValue("WEIGHT FUNCTION:",0,
				  "Characteristics of the k-space weight function");
      parameters.addMember("weightKSigma", &weightKSigma, def | lowopen,
			   "Length scale in the k-space weight", 1.35, 0.);
      parameters.addMember("weightN", &weightN, def | low,
			   "Power-law index of the k-space weight", 4, 2);


      parameters.addMemberNoValue("PSF:",0,
				  "Information on the targets that will be measured");
      parameters.addMember("targetPsf",&targetPsfFile, def,
			   "FITS file holding the PSF for the targets",
			   "starfield_image-000-0.fits");
      parameters.addMember("targetPsfStampSize",&targetPsfStampSize, def | low,
			   "Pixels in each target PSF postage stamp", 48, 16);
      parameters.addMember("targetPsfNx",&targetPsfNx, def | low,
			   "x index of desired target PSF image (0=leftmost)", 0, 0);
      parameters.addMember("targetPsfNy",&targetPsfNy, def | low,
			   "y index of desired target PSF image (0=lowest)", 0, 0);
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

    cout << "#" << stringstuff::taggedCommandLine(argc,argv) << endl;
    parameters.dump(cout);

    // First acquire the PSF;
    fft::SincInterpolant interp1d;
    fft::InterpolantXY interp2d(interp1d);

    double hlr;
    SbPsf targetPsf(Great3Psf(targetPsfFile, interp2d, 
			      targetPsfNx, targetPsfNy,
			      targetPsfStampSize,
			      hlr));
    cout << "# Target PSF hlr, kMax = " << hlr
	 << " " << targetPsf.kMax() << endl;

    // Make the weight function
    KSigmaWeight kw(weightKSigma, weightN);

    // Get list of target images
    list<string> targetFits = stringstuff::fileGlob(targetFiles);
    if (targetFits.empty()) {
      cerr << "No target image files found to match " << targetFiles << endl;
      exit(1);
    }

    // *** Covariances are all the same in this code since PSF, noise, and weight are fixed.
    MomentCovariance<UseMoments> cov;
    bool firstGalaxy = true;

    vector<Moments<UseMoments> > mvec;
    vector<double> vx;
    vector<double> vy;

    // Loop through target images to measure
    for (list<string>::const_iterator iFile = targetFits.begin();
	 iFile != targetFits.end();
	 ++iFile) {
      cerr << "# Working on " << *iFile << endl;
      img::FitsImage<> fi(*iFile, FITS::ReadOnly, 0);

      // Loop through all postage stamps in the file, collecting their
      // moments.  

      cerr << "# Measuring galaxies " << endl;
      for (int iy0 = 1; iy0<fi.getBounds().getYMax(); iy0+=targetStampSize) {
	for (int ix0 = 1; ix0<fi.getBounds().getXMax(); ix0+=targetStampSize) {
	  Bounds<int> b(ix0,ix0+targetStampSize-1,
			iy0,iy0+targetStampSize-1);
	  // !! change to using subimages and extract image at once, better multithreading?
	  const img::Image<> stamp = fi.use(b);

	  try {
	    double cx, cy;
	    PixelGalaxy<UseMoments> pg = Great3Galaxy<UseMoments>(stamp, 
								  targetPsf, 
								  kw, 
								  targetNoise,
								  cx,cy);

	    // Use first galaxy for covariance matrix
	    if (firstGalaxy) {
	      cov = pg.getCov();
	      firstGalaxy = false;}

	    double dx, dy;
	  
	    TemplateGalaxy<>* shifted = newtonShift(pg, dx, dy, 2);
	    /**/ if (shifted->getMoments()[0]!=shifted->getMoments()[0])
	      cerr << ix0 << " " << iy0 << " " << cx << " " << cy
		   << " " << dx << " " << dy << pg.getMoments() << endl;

	    // Enter shifted moments into our collection:
	    mvec.push_back(shifted->getMoments());
	    vx.push_back(cx + dx);
	    vy.push_back(cy + dy);

	    /**/if (ix0==1) cerr << "center shift: " << cx << " " << cy
				 << " Moments " << shifted->getMoments()[MI::CX]
				 << " " << shifted->getMoments()[MI::CY]
				 << endl;
	    delete shifted;
	  } catch (std::runtime_error& e) {
	    cerr << "Measurement error for galaxy at " << ix0 << ", " << iy0 << ", skipping it..." << endl;
	  }
	} // end ix loop
      } // iy loop
    } // end file loop

    // Now produce the output FITS file.
    // Primary extension is a small image giving the covariance matrix
    img::Image<float> covimg(MI::N, MI::N);
    for (int i=0; i<MI::N; i++)
      for (int j=0; j<MI::N; j++)
	covimg(i+1,j+1) = cov(i,j);
    covimg.shift(1,1);

    // Write into the header which moments are in use
    bool b = MI::UseFlux;
    covimg.header()->append("USE_FLUX", b, "Flux moments in use?");
    b = MI::UseCentroid;
    covimg.header()->append("USE_CENT", b, "Centroid moments in use?");
    b = MI::UseE;
    covimg.header()->append("USE_E", b, "E moments in use?");
    b = MI::UseSize;
    covimg.header()->append("USE_SIZE", b, "Size moments in use?");

    {
      img::FitsImage<float> fi(outfile, 
			       FITS::ReadWrite + FITS::Create + FITS::OverwriteFile,
			       0);
      fi.copy(covimg);
    }
    
    // Next make a table holding positions and moments and an empty Pqr for each galaxy
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

    // Zeros for Pqr
    {
      vector<float> vm(Pqr::SIZE,0.);
      vector<vector<float> > vvf;
      for (long i=0; i<nRows; i++) 
	vvf.push_back(vm);
      tab.addColumn(vvf, "PQR", Pqr::SIZE);
    }

    // Write to FITS
    FITS::FitsTable ft(outfile, FITS::ReadWrite + FITS::Create, 1);
    ft.copy(tab);

  } catch (std::runtime_error& e) {
    quit(e,1);
  }
}


