// Class that implements the Psf interface for any SBProfile
#ifndef SBPSF_H
#define SBPSF_H

#include "PixelGalaxy.h"
#include "SBProfile.h"
#include "Random.h"

namespace bfd {
  class SbPsf: public Psf {
  public:
    // When creating the SbPsf, kmax will be set to smallest
    // |k| for which |kValue|<threshold*flux, or
    // to kMax() of the sbp, whichever is lower.
    // Will store a duplicate of the input sb
    SbPsf(const sbp::SBProfile& sb_, double threshold=1e-4): sb(sb_.duplicate()),
							     kmax(sb->maxK()) {
      double dk = sb->stepK();
      img::Image<> re;
      img::Image<> im;
      sb->drawK(re, im, dk);
      re = re*re + im*im;
      double lowerLimit = pow(threshold*sb->getFlux(), 2.);
      for (int iy=re.yMin(); iy<=re.yMax(); iy++) {
	double kysq = iy*dk*iy*dk;
	for (int ix=0; ix<=re.xMax(); ix++) {
	  if (re(ix,iy) < lowerLimit) {
	    kmax = MIN(kmax, sqrt(kysq+ix*dk*ix*dk));
	  }
	}
      }
    }
    ~SbPsf() {delete sb;}
    SbPsf(const SbPsf& rhs): sb(rhs.sb->duplicate()), kmax(rhs.kmax) {}
    const SbPsf& operator=(const SbPsf& rhs) {
      if (this!=&rhs) {
	delete sb;
	sb = rhs.sb->duplicate();
	kmax = rhs.kmax;
      }
      return *this;
    }

    virtual DComplex kValue(double kx, double ky) const {
      return sb->kValue(Position<double>(kx,ky));
    }
    virtual double kMax() const {return kmax;}
  private:

    const sbp::SBProfile* sb;
    double kmax;
  };

  // Function that will build a PixelGalaxy by obtaining information in k space directly from
  // a SBProfile galaxy.  Assumes galaxy is *not* convolved with PSF yet, but noise will be
  // calculated assuming that it would be convolved, observed w/noise, then deconvolved.
  template<int UseMoments>
  extern
  PixelGalaxy<UseMoments>*
  SbGalaxy(const KWeight& kw,
	   const sbp::SBProfile& sbg,
	   const Psf& psf,
	   ran::UniformDeviate& ud,
	   double noise=1.,
	   bool addNoise = true);

} // namespace bfd

#endif // SBPSF_H
