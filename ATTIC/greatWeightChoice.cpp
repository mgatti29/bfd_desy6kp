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

string usage = "greatWeightChoice: Find noise, PSF size, and best weight size for \n"
  "   a series of GREAT3 target images/psfs.\n"
  "usage: greatWeightChoice [parameter file] [-parameter [=] value...]\n"
  "       Parameter files read in order of command line, then command-line parameters,\n"
  "       later values override earlier.\n"
  "stdin: none\n"
  "     Run with -help to get parameter list\n"
  "stdout: table of information on each input exposure";

using namespace bfd;

int main(int argc,
	 char *argv[])
{
  const int UseMoments=USE_ALL_MOMENTS;
  typedef MomentIndices<UseMoments> MI;

  double sigmaMin;
  double sigmaMax;
  double sigmaStep;
  int weightN;

  int nxPsf;
  int nyPsf;
  int psfStampSize;

  string templateFile;
  string templatePsfFile;
  int galaxyStampSize;

  string targetFiles;
  string psfPrefix;

  cout << "#" << stringstuff::taggedCommandLine(argc,argv) << endl;

  try {

    Pset parameters;
    {
      const int def=PsetMember::hasDefault;
      const int low=PsetMember::hasLowerBound;
      const int up=PsetMember::hasUpperBound;
      const int lowopen = low | PsetMember::openLowerBound;
      const int upopen = up | PsetMember::openUpperBound;

      parameters.addMember("targetFiles",&targetFiles, def,
			   "String(s) to glob to get target image filenames", 
			   "image*fits");
      parameters.addMember("psfPrefix",&psfPrefix, def,
			   "String to prepend to target filenames for matching PSF",
			   "starfield_");
      parameters.addMember("templateFile",&templateFile, def,
			   "Template image filename", 
			   "deep_image-000-0.fits");
      parameters.addMember("templatePsfFile",&templatePsfFile, def,
			   "Template image PSF filename", 
			   "deep_starfield_image-000-0.fits");


      parameters.addMember("sigmaMin", &sigmaMin, def | lowopen,
			   "Start of sigma scan", 0.5, 0.);
      parameters.addMember("sigmaMax", &sigmaMax, def | lowopen,
			   "End of sigma scan", 1.7, 0.);
      parameters.addMember("sigmaStep", &sigmaStep, def | lowopen,
			   "Step of sigma scan", 0.1, 0.);
      parameters.addMember("weightN", &weightN, def | low,
			   "Power-law index of the k-space weight", 4, 2);

      parameters.addMember("galaxyStampSize",&galaxyStampSize, def | low,
			   "Pixels in each galaxy postage stamp", 48, 16);
      parameters.addMember("psfStampSize",&psfStampSize, def | low,
			   "Pixels in each PSF postage stamp", 48, 16);
      parameters.addMember("nxPsf", &nxPsf, def | low,
			   "PSF x index", 0, 0);
      parameters.addMember("nyPsf", &nyPsf, def | low,
			   "PSF y index", 0, 0);

    }

    parameters.setDefault();
  
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


    //  Get a list of the target files
    list<string> targetList;
    {
      list<string> globs = stringstuff::split(targetFiles,',');
      for (list<string>::iterator i = globs.begin(); i!=globs.end(); ++i) {
	stringstuff::stripWhite(*i);
	list<string> tmp = stringstuff::fileGlob(*i);
	targetList.splice(targetList.end(), tmp);
      }
    }

    if (targetList.empty()) {
      cerr << "No target files found to match " << targetFiles << endl;
      exit(1);
    }

    // Acquire the template PSF;
    fft::SincInterpolant interp1d;
    fft::InterpolantXY interp2d(interp1d);

    double hlr;
    SbPsf templatePsf(Great3Psf(templatePsfFile, interp2d, nxPsf, nyPsf, psfStampSize, hlr));

    img::FitsImage<> fi(templateFile, FITS::ReadOnly, 0);
    // Loop through all postage stamps in the file

    vector<double> dMdE;

    for (double sigma = sigmaMin; sigma <= sigmaMax; sigma+=sigmaStep) {

      // Make the weight function
      KSigmaWeight kw(sigma, weightN);

      double sumdm1sq = 0.;
      double sumdm2sq = 0.;
      int nGals = 0;

      for (int iy0 = 1; iy0<fi.getBounds().getYMax(); iy0+=galaxyStampSize) {
	for (int ix0 = 1; ix0<fi.getBounds().getXMax(); ix0+=galaxyStampSize) {
	  Bounds<int> b(ix0,ix0+galaxyStampSize-1,
			iy0,iy0+galaxyStampSize-1);
	  const img::Image<> stamp = fi.use(b);

	  PixelGalaxy<UseMoments> pg = Great3Galaxy<USE_ALL_MOMENTS>(stamp, 
								     templatePsf, kw, 1.);
	  // Do a Newton iteration on the centroid
	  Moments<UseMoments> m = pg.getMoments();
	  DVector2 dm;
	  dm[0] = m[MI::CX];
	  dm[1] = m[MI::CY];
	  DMatrix22 dmdx;
	  dmdx(0,0) = pg.dMdx()[MI::CX];
	  dmdx(1,0) = pg.dMdx()[MI::CY];
	  dmdx(0,1) = pg.dMdy()[MI::CX];
	  dmdx(1,1) = pg.dMdy()[MI::CY];
	  dm /= dmdx;

	  TemplateGalaxy<UseMoments>* shifted =
	    dynamic_cast<TemplateGalaxy<UseMoments>*> (pg.getShifted(-dm[0],-dm[1]));

	  MomentDerivs<UseMoments> deriv = shifted->getDerivs();
	  sumdm1sq += SQR(deriv(MI::E1,Pqr::DG1));
	  sumdm2sq += SQR(deriv(MI::E2,Pqr::DG2));
	  nGals++;
	  delete shifted;
	}
      }
      dMdE.push_back( (sumdm1sq + sumdm2sq) / (2.*nGals));
      /**/cerr << "sigma " << sigma << " dmde " << dMdE.back() << endl;
    }

    // Loop through the target files
    for (list<string>::iterator iFile = targetList.begin();
	 iFile != targetList.end();
	 ++iFile) {
      string targetFile = *iFile;

      img::FitsImage<> fi(targetFile, FITS::ReadOnly, 0);

      // Measure the noise
      double noise;
      {
	const img::Image<> all = fi.use();
	vector<double> pix;
	const int boxrad = 3;  // half-side of box to use at each stamp corner

	for (int iy0 = galaxyStampSize +1; 
	     iy0<fi.getBounds().getYMax(); 
	     iy0+=galaxyStampSize) {
	  for (int ix0 = galaxyStampSize +1; 
	       ix0<fi.getBounds().getXMax(); 
	       ix0+=galaxyStampSize) {
	    for (int iy=iy0-boxrad; iy<iy0+boxrad; iy++)
	      for (int ix=ix0-boxrad; ix<ix0+boxrad; ix++)
		pix.push_back(all(ix,iy));
	  }
	}
	stats::pctileClip(pix, 5.);
	noise = stats::variance(pix);
      }

      // Get the PSF

      double hlr;
      SbPsf psf(Great3Psf(psfPrefix + targetFile, interp2d, 0, 0, psfStampSize, hlr));

      // And get a chunk of image.
      Bounds<int> b(1, galaxyStampSize,
		    1, galaxyStampSize);
      const img::Image<> stamp = fi.use(b);

      // Find weight sigma giving best S/N.
      double bestSigma=0.;
      double bestVar=0.;
      double bestFom=0.;

      // Loop through all postage stamps in the file

      cout << "# TargetFile    PSF_HLR    PixVariance   sigmaW   EVar   EFoM " << endl;
      for (int isigma=0; isigma<dMdE.size(); isigma++) {
	double sigma = sigmaMin + isigma*sigmaStep;
	// Get the covariance matrix at each weight sigma.

	// Make the weight function
	KSigmaWeight kw(sigma, weightN);

	PixelGalaxy<UseMoments> pg = Great3Galaxy<USE_ALL_MOMENTS>(stamp, 
								   psf, kw, noise);

	MomentCovariance<UseMoments> cov = pg.getCov();
	double var = 0.5*(cov(MI::E1,MI::E1) + cov(MI::E2,MI::E2));
	double fom = dMdE[isigma] / var;
	
	if (fom > bestFom) {
	  bestFom = fom;
	  bestSigma = sigma;
	  bestVar = var;
	}
      }

      cout << targetFile
	   << " " << hlr
	   << " " << noise
	   << " " << bestSigma
	   << " " << bestVar
	   << " " << bestFom
	   << endl;
    } // end file loop

  } catch (std::runtime_error& e) {
    quit(e,1);
  }
}


