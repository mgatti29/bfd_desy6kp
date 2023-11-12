// Integrate a collection of galaxy moments over prior, accumulate to its Pqr in the file
#include <fstream>

#include "StringStuff.h"
#include "Pset.h"
#include "Prior.h"
#include "KdPrior.h"
#include "fft.h"
#include "Interpolant.h"
#include "Image.h"
#include "FTable.h"
#include "FitsImage.h"
#include "FitsTable.h"
#include "Distributions.h"
#include "Great3.h"
#include "Selection.h"
#include "Stopwatch.h"

string usage = "greatIntegrate: Make a prior from a template image file and integrate\n"
  " target moments over this prior.  Pqr values in the input moment files are incremented\n"
  " by the results of the integration.\n"
  "usage: greatIntegrate [parameter file] [-parameter [=] value...]\n"
  "       Parameter files read in order of command line, then command-line parameters,\n"
  "       later values override earlier.\n"
  "stdin: none\n"
  "     Run with -help to get parameter list\n"
  "     momentFiles parameter is comma-separated list of moment FITS file names.\n"
  "      Each can be a glob.\n"
  "     Likewise the prior can be constructed from union of images in a set of templateFits\n"
  "      files.  Number of templatePSF images should be 1 or match # of template images.\n"
  "     doDeselection = true will calculate PQR contribution only for objects that do\n"
  "      *not* make the flux cut.\n"
  "stdout: Diagnostic information";

using namespace bfd;

const int UseMoments=USE_ALL_MOMENTS;
typedef MomentIndices<UseMoments> MI;

MomentCovariance<UseMoments>
covarianceFromFits(string momentFile) {
  // Function to extract covariance matrix from the primary extension image of FIT file
  img::FitsImage<float> fi(momentFile, FITS::ReadOnly, 0);
  img::Image<float> covimg = fi.extract();
  if (covimg.xMax() != MI::N || covimg.yMax() != MI::N) {
    const int N = MI::N;  // This declaration needed to avoid icpc problem
    FormatAndThrow<std::runtime_error>() << "Restoring moments from file "
					 << momentFile
					 << " has wrong covariance image size, want "
					 << N
					 << " but have " << covimg.xMax()
					 << " x " << covimg.yMax();
  }
  bool test;

  if (!covimg.header()->getValue("USE_FLUX",test) || test != MI::UseFlux)
    FormatAndThrow<std::runtime_error>() << "Restoring moments from file "
					 << momentFile
					 << " has wrong/missing USE_FLUX";
  if (!covimg.header()->getValue("USE_CENT",test) || test != MI::UseCentroid)
    FormatAndThrow<std::runtime_error>() << "Restoring moments from file "
					 << momentFile
					 << " has wrong/missing USE_CENT";
  if (!covimg.header()->getValue("USE_E",test) || test != MI::UseE)
    FormatAndThrow<std::runtime_error>() << "Restoring moments from file "
					 << momentFile
					 << " has wrong/missing USE_E";
  if (!covimg.header()->getValue("USE_SIZE",test) || test != MI::UseSize)
    FormatAndThrow<std::runtime_error>() << "Restoring moments from file "
					 << momentFile
					 << " has wrong/missing USE_SIZE";
  MomentCovariance<UseMoments> cov;
  for (int i=0; i<MI::N; i++)
    for (int j=0; j<MI::N; j++) 
      cov(i,j) = covimg(i+1,j+1);

  return cov;
}

int main(int argc,
	 char *argv[])
{
  string templateFiles;
  int templateStampSize;

  string templatePsfFiles;
  int templatePsfNx;
  int templatePsfNy;
  int templatePsfStampSize;

  double weightKSigma;
  int weightN;

  string nominalMomentFile;
  double priorSigmaCutoff;
  double priorSigmaStep;
  double priorSigmaBuffer;

  string momentFiles;
  int nSample;

  double noiseFactor;
  double selectSnMin;
  double selectSnMax;
  bool   doDeselection;

  Pset parameters;
  {
      const int def=PsetMember::hasDefault;
      const int low=PsetMember::hasLowerBound;
      const int up=PsetMember::hasUpperBound;
      const int lowopen = low | PsetMember::openLowerBound;
      const int upopen = up | PsetMember::openUpperBound;

      parameters.addMemberNoValue("TEMPLATES:",0,
				  "The template image files");
      parameters.addMember("templateFiles",&templateFiles, def,
			   "Template image filenames", 
			   "deep_image*fits");
      parameters.addMember("templateStampSize",&templateStampSize, def | low,
			   "Pixels in each template postage stamp", 48, 16);

      parameters.addMemberNoValue("TEMPLATE PSF:",0,
				  "The PSF for the template images");
      parameters.addMember("templatePsfs",&templatePsfFiles, def,
			   "FITS file(s) holding the PSF for the template images",
			   "deep_starfield_image-000-0.fits");
      parameters.addMember("templatePsfStampSize",&templatePsfStampSize, def | low,
			   "Pixels in each template PSF postage stamp", 48, 16);
      parameters.addMember("templatePsfNx",&templatePsfNx, def | low,
			   "x index of desired template PSF image (0=leftmost)", 0, 0);
      parameters.addMember("templatePsfNy",&templatePsfNy, def | low,
			   "y index of desired template PSF image (0=lowest)", 0, 0);

      parameters.addMemberNoValue("WEIGHT FUNCTION:",0,
				  "Characteristics of the k-space weight function");
      parameters.addMember("weightKSigma", &weightKSigma, def | lowopen,
			   "Length scale in the k-space weight", 1.0, 0.);
      parameters.addMember("weightN", &weightN, def | low,
			   "Power-law index of the k-space weight", 4, 2);

      parameters.addMemberNoValue("PRIOR CONFIGURATION:",0,
				  "Characteristics of the sampled prior");
      parameters.addMember("nominalMomentFile",&nominalMomentFile, def,
			   "FITS moment file holding nominal covariance",
			   "moments_000.fits");
      parameters.addMember("priorSigmaCutoff",&priorSigmaCutoff, def | low,
			   "Maximum sigma range when sampling for prior", 5.5, 3.);
      parameters.addMember("priorSigmaStep",&priorSigmaStep, def | lowopen,
			   "Step size when sampling for prior", 1., 0.);
      parameters.addMember("priorSigmaBuffer",&priorSigmaBuffer, def | low,
			   "Buffer width of KdTreePrior (in sigma)", 1., 0.);

      parameters.addMemberNoValue("TARGETS:",0,
				  "Information on the targets that will be measured");
      parameters.addMember("momentFiles",&momentFiles, def,
			   "Globs for files with target moment FITS table and Pqr's",
			   "moments.fits");
      parameters.addMember("nSample",&nSample, def | low,
			   "Number of templates sampled per target (0=all)", 50000, 0);
      parameters.addMember("noiseFactor",&noiseFactor, def | low,
			   "Noise amplification factor for moments(if >1)", 0., 0.);
      parameters.addMember("selectSnMin",&selectSnMin, def | low,
			   "Nominal flux S/N of lower flux selection threshold", 5., 0.);
      parameters.addMember("selectSnMax",&selectSnMax, def | low,
			   "Nominal flux S/N of upper flux selection threshold", 0., 0.);
      parameters.addMember("doDeselection",&doDeselection, def,
			   "Calculate PQR only for galaxies failing flux selection", false);
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


    // Get the list of moment files
    list<string> momentList;
    {
      list<string> globs = stringstuff::split(momentFiles,',');
      for (list<string>::iterator i = globs.begin(); i!=globs.end(); ++i) {
	stringstuff::stripWhite(*i);
	list<string> tmp = stringstuff::fileGlob(*i);
	momentList.splice(momentList.end(), tmp);
      }
    }

    if (momentList.empty()) {
      cerr << "No moment files found to match " << momentFiles << endl;
      exit(1);
    }

    // Get the list of template images
    list<string> templateList;
    {
      list<string> globs = stringstuff::split(templateFiles,',');
      for (list<string>::iterator i = globs.begin(); i!=globs.end(); ++i) {
	stringstuff::stripWhite(*i);
	list<string> tmp = stringstuff::fileGlob(*i);
	templateList.splice(templateList.end(), tmp);
      }
    }

    if (templateList.empty()) {
      cerr << "No template files found to match " << templateFiles << endl;
      exit(1);
    }

    // Get the list of template PSF images
    list<string> templatePsfList;
    {
      list<string> globs = stringstuff::split(templatePsfFiles,',');
      for (list<string>::iterator i = globs.begin(); i!=globs.end(); ++i) {
	stringstuff::stripWhite(*i);
	list<string> tmp = stringstuff::fileGlob(*i);
	templatePsfList.splice(templatePsfList.end(), tmp);
      }
    }

    if (templatePsfList.size()!= 1 && templatePsfList.size()!=templateList.size()) {
      cerr << "Number of template Psf files is neither 1"
	" nor matches number of template PSF files:" << templateFiles << endl;
      exit(1);
    }

    ran::UniformDeviate ud;
    ran::GaussianDeviate gd(ud);

    // Make the weight function
    KSigmaWeight kw(weightKSigma, weightN);

    // Acquire the template PSF;
    fft::SincInterpolant interp1d;
    fft::InterpolantXY interp2d(interp1d);

    // Get the nominal covariance matrix from the nominal moments file:
    MomentCovariance<UseMoments> cov = covarianceFromFits(nominalMomentFile);

    double fluxNoise = sqrt(cov(MI::FLUX,MI::FLUX));
    double fluxMin = fluxNoise * selectSnMin;
    double fluxMax = fluxNoise * selectSnMax;

    // Create KdTree prior, use all samples if nSample=0
    KDTreePrior<UseMoments> prior(fluxMin, fluxMax, false,
				  nSample, ud, priorSigmaBuffer,MID,
				  nSample<=0);
    prior.setNominalCovariance(cov);
    prior.setSamplingRange(priorSigmaCutoff, priorSigmaStep);
    prior.setUniqueCeiling(200);  // Stop counting unique templates above 200.
    // If we only need selection probabilities, do this to save memory of rotated copies:
    if (doDeselection) prior.setSelectionOnly();
    // Add noise to moments if desired
    if (noiseFactor > 1.)
      prior.addNoiseFactor(noiseFactor, ud);

    const double maxXY = 5.;	// Centroid error stops here

    // Start timer for template acquisition
    Stopwatch timer;
    timer.start();

    int templateID = 0;
    // Loop through template images and postage stamps to build prior
    while (!templateList.empty()) {
      
      // Get the PSF
      string templatePsfFile = templatePsfList.front();
      if (templatePsfList.size()>1) templatePsfList.pop_front();
      double hlr;
      SbPsf templatePsf(Great3Psf(templatePsfFile, interp2d, 
				  templatePsfNx, templatePsfNy,
				  templatePsfStampSize,
				  hlr));
      cout << "# Template PSF hlr, kMax = " << hlr
	   << " " << templatePsf.kMax() << endl;

      string templateFile = templateList.front();
      templateList.pop_front();
      cerr << "# Ingesting templates from " << templateFile << endl;
      // Make a list of all postage stamps
      vector<img::Image<> > vStamps;
      {
	img::FitsImage<> fi(templateFile, FITS::ReadOnly, 0);
	// Loop through all postage stamps in the file
	for (int iy0 = 1; iy0<fi.getBounds().getYMax(); iy0+=templateStampSize) {
	  for (int ix0 = 1; ix0<fi.getBounds().getXMax(); ix0+=templateStampSize) {
	    vStamps.push_back(fi.extract(Bounds<int>(ix0,ix0+templateStampSize-1,
						     iy0,iy0+templateStampSize-1)));
	  }
	}
      }
      // Acquire template images in a loop, parallelizing if possible
#ifdef _OPENMPXX
#pragma omp parallel
#endif
      {
#ifdef _OPENMPXX
#pragma omp for schedule(dynamic)
#endif
	for (int iStamp=0; iStamp<vStamps.size(); iStamp++) {
	  PixelGalaxy<UseMoments> pg = Great3Galaxy<UseMoments>(vStamps[iStamp],
								templatePsf, 
								kw, 
								1.);
	  prior.addTemplate(pg, ud, maxXY, 1., true, templateID + iStamp);
	}
      } // end parallel block
      templateID += vStamps.size();	// Move up the id pointer for next file
    } // end input file loop

    timer.stop();
    cout << "# Created " << prior.getNTemplates() << " templates in "
	 << timer << endl;
    timer.reset();

    cerr << "Building KdTree for " << prior.getNTemplates() << " templates..." << endl;
    timer.start();
    // Prepare the prior (builds tree, etc.)
    prior.prepare();
    timer.stop();
    cout << "# prepared in " << timer << endl;
    timer.reset();

    // Now enter a loop to integrate each of the moment files against the prior
    timer.start();

    while (!momentList.empty()) {
      string momentFile = momentList.front();
      momentList.pop_front();
      cerr << "Working on input file " << momentFile << endl;
      MomentCovariance<UseMoments> targetCov = covarianceFromFits(momentFile);

      // Collect indices and moments of the galaxies we want to measure
      FITS::FitsTable ft(momentFile, FITS::ReadWrite, 1);
      img::FTable tab = ft.use();

      {
	// Add columns for template count and unique count if not present
	vector<string> colList = tab.listColumns();
	bool hasTemplates = false;
	bool hasUnique = false;
	for (vector<string>::iterator i = colList.begin();
	     i != colList.end();
	     ++i) {
	  if (stringstuff::nocaseEqual(*i, "NTEMPLATE")) hasTemplates=true;
	  if (stringstuff::nocaseEqual(*i, "NUNIQUE")) hasUnique=true;
	}
	if (!hasTemplates) {
	  vector<int> vi(1,0);
	  tab.addColumn(vi, "NTEMPLATE");
	}
	if (!hasUnique) {
	  vector<int> vi(1,0);
	  tab.addColumn(vi, "NUNIQUE");
	}
      }
      // Vector of row numbers of targets we will analyze here.
      vector<int> useRows;
      vector<MI::MVector> useMoments;
      for (int irow=0; irow<tab.nrows(); irow++) {
	MI::MVector mv;
	vector<float> vf(MI::N);
	tab.readCell(vf, "MOMENTS", irow);
	for (int i=0; i<MI::N; i++) mv[i] = vf[i];
	bool selected = (fluxMin==0 || mv[MI::FLUX]>fluxMin)
	  && (fluxMax==0. || mv[MI::FLUX]<=fluxMax);
	if ( doDeselection ^ selected ) {
	  // Measure this galaxy, if it's appropriately inside / outside selection region
	  useRows.push_back(irow);
	  useMoments.push_back(mv);
	}
      }
      const int nTarget = useRows.size();

      if (doDeselection) {
	// All we will do in this case is put the deselection PQR into each row
	// for objects missing the flux cut.
	Pqr pSelect = prior.selectionProbability(targetCov);
	vector<float> vf(Pqr::SIZE);
	// Negative values for nTemplate, nUnique will signal deselection
	int n = -1;
	// Change the probability of selection into probability of NOT being selected
	for (int j=0; j<Pqr::SIZE; j++) vf[j] = -pSelect[j];
	vf[Pqr::P] = 1.-pSelect[Pqr::P];
	for (int j=0; j < nTarget; j++) {
	  tab.writeCell(vf,"PQR",useRows[j]);
	  tab.writeCell(n, "NTEMPLATE", useRows[j]);
	  tab.writeCell(n, "NUNIQUE", useRows[j]);
	}
	// **** Done with this input file.  Skip the integration and move to next
	continue;
      }

      // Now we'll distribute the integrations over processors
      cout << "# " << nTarget << " targets in " << momentFile << endl;
      const int chunk = MIN(100, nTarget);	//Number of galaxies to measure per for loop
      int nLoops = (nTarget-1) / chunk + 1;

      vector<Pqr> vpqr(nTarget);
      vector<int> vnTemp(nTarget);
      vector<int> vnUnique(nTarget);

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
	for (int i=0; i<nLoops; i++) {
	  int subN=0;
	  int jstart = i*chunk;
	  int jend = MIN((i+1)*chunk,nTarget);
	  int nTemplates;
	  int nUnique;
	  for (int j=jstart; j<jend; j++) {
	    if (j%1000==0) cerr << "Doing galaxy " << j << endl;

	    // The integration happens here:
	    vpqr[j] = prior.getPqr2(useMoments[j],targetCov,nTemplates,nUnique);
	    vnTemp[j] = nTemplates;
	    vnUnique[j] = nUnique; 
	  }
	} // end galaxy loop
      } // end parallel block

      // Add the results into the Pqr table column, and the template and unique counts
      for (int j=0; j<nTarget; j++) {
	vector<float> vf;
	tab.readCell(vf, "PQR", useRows[j]);
	for (int i=0; i<Pqr::SIZE; i++) vf[i] += vpqr[j][i];
	tab.writeCell(vf,"PQR",useRows[j]);
	int n;
	tab.readCell(n, "NTEMPLATE", useRows[j]);
	n += vnTemp[j];
	tab.writeCell(n, "NTEMPLATE", useRows[j]);
	tab.readCell(n, "NUNIQUE", useRows[j]);
	n += vnUnique[j];
	tab.writeCell(n, "NUNIQUE", useRows[j]);
      }

    } // end input file loop.
    timer.stop();
    cout << "# integration in " << timer << endl;

  } catch (std::runtime_error& e) {
    quit(e,1);
  }
}


