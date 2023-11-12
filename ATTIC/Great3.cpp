// Routines to mate the GREAT3 Challenge data to our bfd codes.

#include "Great3.h"
#include "FitsImage.h"
#include "SBPixel.h"
#include <map>

using namespace bfd;

SbPsf
bfd::Great3Psf(const string fitsname, 
	       const fft::Interpolant2d& xInterp,
	       int nx, int ny, 
	       int stampSize,
	       double& halfLightRadius) {
  img::FitsImage<> fi(fitsname, FITS::ReadOnly, 0);
  Bounds<int> b(nx*stampSize+1, (nx+1)*stampSize, ny*stampSize+1, (ny+1)*stampSize);
  const img::Image<> data = fi.use(b);

  double x0 = 0.5*(data.xMin()+data.xMax());
  double y0 = 0.5*(data.yMin()+data.yMax());

  double sigma = stampSize / 12.;
  centroid(data, sigma, x0, y0, halfLightRadius);

  // Try again with an adjusted sigma if it was very big or small compared to half-light radius:
  if (sigma < 0.8*halfLightRadius || sigma > 1.5*halfLightRadius) {
    sigma = 1.2*halfLightRadius;
    centroid(data, sigma, x0, y0, halfLightRadius);
  }

  // Make an SBProfile from the stamp
  sbp::SBPixel pixPsf(data, xInterp);

  pixPsf.setFlux(1.);
  // The pixel value where the FFT is placing its origin is 1 past halfway:
  int xOriginFT = (data.xMin() + data.xMax() + 1)/2;
  int yOriginFT = (data.yMin() + data.yMax() + 1)/2;

  // Change FT phases to move centroid back to the FFT phase center
  sbp::SBProfile* shifted = pixPsf.shift(xOriginFT-x0, yOriginFT-y0);
  SbPsf out(*shifted);
  delete shifted;
  return out;
}

// Return a PixelGalaxy from a postage stamp contained within an image
// Will use the specified PSF and weight function.
// Sigma is size to use for centroiding.
template<int UseMoments>
PixelGalaxy<UseMoments>
bfd::Great3Galaxy(const img::Image<> stamp, const Psf& psf, const KWeight& kw, double noise) {
  double cx, cy;
  return Great3Galaxy<UseMoments>(stamp, psf, kw, noise,cx,cy);
}


// This one returns the coordinates of the phase center used
template<int UseMoments>
PixelGalaxy<UseMoments>
bfd::Great3Galaxy(const img::Image<> stamp, const Psf& psf, const KWeight& kw, double noise,
		  double& cx, double& cy) {

  // First determine a centroid for the target:
  double x0 = 0.5*(stamp.xMin()+stamp.xMax());
  double y0 = 0.5*(stamp.yMin()+stamp.yMax());
  double sigma = (stamp.xMax() - stamp.xMin() + 1) / 12.;
  double halfLightRadius;
  centroid(stamp, sigma, x0, y0, halfLightRadius);

  // Try again with an adjusted sigma if it was very big or small compared to half-light radius:
  if (sigma < 0.8*halfLightRadius || sigma > 1.5*halfLightRadius) {
    sigma = 1.2*halfLightRadius;
    centroid(stamp, sigma, x0, y0, halfLightRadius);
  }

  DVector2 origin;
  origin[0] = x0;
  origin[1] = y0;
  cx = x0;
  cy = y0;
  Affine shift(origin);
  return PixelGalaxy<UseMoments>(kw, stamp, shift, psf, noise);
}


void
bfd::centroid(const img::Image<> data, 
	      double sigma, 
	      double& x0, double& y0, 
	      double& halfLightRadius) {
  const int nIterations=4; // *** note fixed number of iterations here.
  Bounds<int> b = data.getBounds();
  if (x0 <= b.getXMin() || x0 >= b.getXMax() || y0<= b.getYMin() || y0>b.getYMax()) {
    // Start origin in center of image if no other start was given
    x0 = 0.5* (b.getXMin() + b.getXMax());
    y0 = 0.5* (b.getYMin() + b.getYMax());
  }
  // Create a multimap that will order the fluxes in radius for us
  std::multimap<double,double> fVsR;
  double factor = -0.5/(sigma*sigma);
  double maxRR = 16.*sigma*sigma;	// aperture for getting hlr
  double flux = 0.;
  double x = data(b.getXMin(), b.getYMin());

  for (int iter=0; iter<nIterations; iter++) {
    fVsR.clear();
    double sumx=0.;
    double sumy=0.;
    double sumw = 0.;
    flux = 0.;
    for (int iy=b.getYMin(); iy<=b.getYMax(); iy++) {
      double dy = iy - y0;
      for (int ix=b.getXMin(); ix<=b.getXMax(); ix++) {
	double dx = ix - x0;
	double rr = dx*dx + dy*dy;
	if (rr<=maxRR) {
	  fVsR.insert(std::pair<double,double>(rr, data(ix,iy)));
	  flux += data(ix,iy);
	}
	double w = exp(rr*factor) * data(ix,iy);
	sumw += w;
	sumx += w*dx;
	sumy += w*dy;
      }
    }
    x0 += sumx / sumw;
    y0 += sumy / sumw;
  }

  // After last iteration, calculate the half-light radius
  double light = 0.;
  double rr = 0.;
  for (multimap<double,double>::const_iterator i=fVsR.begin();
       i != fVsR.end();
       ++i) {
    double dl = i->second;
    if (light + dl > 0.5*flux) {
      // We have crossed the half-light radius.  Interpolate.
      double fraction = (0.5*flux - light) / dl;
      halfLightRadius = sqrt(rr) + fraction*(sqrt(i->first) - sqrt(rr));
      return;
    } else {
      light += dl;
      rr = i->first;
    }
  }
  // Should not get through all the pixels with triggering the half-light threshold.
  throw std::runtime_error("Great3::centroid() did not get above half of flux for hlr");
}

void
bfd::centroid(const img::Image<> data, double sigma, double& x0, double& y0) {
  double hlr;
  centroid(data, sigma, x0, y0, hlr);
}


//template<int UseMoments>
TemplateGalaxy<>*
bfd::newtonShift(const TemplateGalaxy<>& gin, 
		 double& cx, double& cy,
		 int iterations=2) {
  const int UseMoments = USE_ALL_MOMENTS;
  typedef MomentIndices<UseMoments> MI;
  cx = cy = 0.;
  if (!MI::UseCentroid) {
    //**if (true) {  
    // Not using centroid, just return a copy of input
    return dynamic_cast<TemplateGalaxy<UseMoments>*> (gin.getShifted(-cx,-cy));
  }
  // Do fixed number of Newton iteration to bring the xy moments near zero:
  const TemplateGalaxy<UseMoments>* gbase=&gin;
  TemplateGalaxy<UseMoments>* shifted=0;
  for (int iter=0; iter<iterations; iter++) {
    DVector2 dm;
    dm[0] = gbase->getMoments()[MI::CX];
    dm[1] = gbase->getMoments()[MI::CY];
    DMatrix22 dmdx;
    dmdx(0,0) = gbase->dMdx()[MI::CX];
    dmdx(0,1) = gbase->dMdy()[MI::CX];
    dmdx(1,0) = gbase->dMdx()[MI::CY];
    dmdx(1,1) = gbase->dMdy()[MI::CY];
  dm /= dmdx;
  cx += dm[0];
  cy += dm[1];
  if (shifted) delete shifted;
  shifted = dynamic_cast<TemplateGalaxy<UseMoments>*> (gin.getShifted(-cx,-cy));
  gbase = shifted;
  /**
  dm[0] = gbase->getMoments()[MI::CX];
  dm[1] = gbase->getMoments()[MI::CY];
  cerr << iter 
	   << " ctr " << cx << " " << cy
	   << " moments " << dm 
	   << endl; 
  /**/
  }
  return shifted;
}

///////////////////////////////////////////////////////////////
// 
// Instantiations of the templates
//
///////////////////////////////////////////////////////////////

#define INSTANTIATE(u) \
  template PixelGalaxy<u> bfd::Great3Galaxy<u>(const img::Image<>,	\
                               const Psf&, const KWeight&, double);  \
  template PixelGalaxy<u> bfd::Great3Galaxy<u>(const img::Image<>,	\
					       const Psf&, const KWeight&, double, \
                                               double& cx, double& cy); \

/**  template TemplateGalaxy<u>*			\
  bfd::newtonShift(const TemplateGalaxy<u>& gin,	\
                   double& cx, double& cy, int iterations);
**/

#include "InstantiateMomentCases.h"
