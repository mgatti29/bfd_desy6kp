// Routines to mate the GREAT3 Challenge data to our bfd codes.
#ifndef GREAT3_H
#define GREAT3_H

#include "Image.h"
#include "Galaxy.h"
#include "PixelGalaxy.h"
#include "SbPsf.h"
#include "KWeight.h"
#include "Interpolant.h"

namespace bfd {

  // Function will return a PSF that is built from one stamp in a FITS file
  // (nx,ny) give the location of the stamp to use.
  extern SbPsf
  Great3Psf(const string fitsname, 
	    const fft::Interpolant2d& interp,
	    int nx, int ny, 
	    int stampSize,
	    double& halfLightRadius);

  // Return a PixelGalaxy from a postage stamp contained within an image
  // Will use the specified PSF and weight function.
  // noise is the variance of pixels.
  template<int UseMoments>
  extern
  PixelGalaxy<UseMoments>
  Great3Galaxy(const img::Image<> stamp, const Psf& psf, const KWeight& kw, double noise);

  // This one returns the coordinates of the phase center used
  template<int UseMoments>
  extern
  PixelGalaxy<UseMoments>
  Great3Galaxy(const img::Image<> stamp, const Psf& psf, const KWeight& kw, double noise,
	       double& cx, double& cy);

  // Function to give a Gaussian-weighted centroid to object near the center
  // of a (sub)image.  Sigma gives the width of weight function.
  // The optional third argument will be filled with the (unweighted) half-light
  // radius of the PSF about the centroid.
  // Input x0, y0 used as starting guess if it's inside the image, else use center of Image.
  extern  void
  centroid(const img::Image<> data, double sigma, double& x0, double& y0, double& halfLightRadius);
  extern void
  centroid(const img::Image<> data, double sigma, double& x0, double& y0);

  // Function that will iterate centroid to try to null the centroid moments
  // Returns a pointer to a new TemplateGalaxy with shifted origin.
  /**  template<int UseMoments>
  extern  TemplateGalaxy<UseMoments>*
  newtonShift(const TemplateGalaxy<UseMoments>& gin, 
  **/
  extern  TemplateGalaxy<>*
  newtonShift(const TemplateGalaxy<>& gin, 
	      double& cx, double& cy,
	      int iterations);
} // namespace bfd

#endif // GREAT3_H
