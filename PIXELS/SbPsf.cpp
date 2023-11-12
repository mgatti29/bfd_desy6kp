// Function to create PixelGalaxy from an SBProfile.
#include "SbPsf.h"
#include "Random.h"

using namespace bfd;

template<int UseMoments>
PixelGalaxy<UseMoments>*
bfd::SbGalaxy(const KWeight& kw,
	      const sbp::SBProfile& sbg,
	      const Psf& psf,
	      ran::UniformDeviate& ud,
	      double noise,
	      bool addNoise) {

  double dk = sbg.stepK();
  double maxK = kw.kMax();
  int Nfft = static_cast<int> (2*floor(maxK/dk));
  if (Nfft < 48) {
    Nfft = 48;
    dk = 2*maxK / Nfft;
  }
  double d2k = dk*dk;
  double kmaxsq = maxK*maxK;
  
  ran::GaussianDeviate gd(ud);
  double sig = 2*PI*sqrt(0.5*noise/d2k);
  
  vector<double> re;
  vector<double> im;
  vector<double> psfre;
  vector<double> psfim;
  vector<double> vkx;
  vector<double> vky;
  int dcIndex = -1;
  
#ifdef _OPENMP
#pragma omp critical(random)
#endif
  {
  for (int iy=-Nfft/2+1; iy<=Nfft/2; iy++) {
    double ky = iy * dk;
    // Do not duplicate the points at ix=0, Nfft/2 in iy<0 half-plane.
    // Their conjugates were already counted in the upper half-plane.
    int ixStart = (iy>=0) ? 0 : 1;
    int ixEnd =  (iy>=0) ? Nfft/2 : Nfft/2-1;
    for (int ix=ixStart; ix<=ixEnd; ix++) {
      double kx = ix*dk;
      if ( kx*kx+ky*ky >= kmaxsq) continue;
      DComplex mtf = psf.kValue(kx,ky);
      DComplex z = sbg.kValue(Position<double>(kx,ky));
      if (iy==0 && ix==0) dcIndex = vkx.size();
      if (addNoise) {
	// Realize noise
	double nr = sig*gd();
	double ni = sig*gd();
	if (iy==0 && ix==0) {
	  // Put all noise into real part.
	  nr += ni;
	  ni = 0.;
	}
	z += DComplex(nr,ni) / std::abs(mtf);
      }

	
      // For the k values that are around the edges of the kTable
      // (at coordinates Nfft/2), we're going to put half of the
      // coefficient at positive ky and half at negative ky so as
      // to symmetrize the k integrals.
      // This should not matter if we properly have the weight function
      // going to zero by kmax of the grid.  But I insert this
      // to make some test cases (like Gaussian weights)
      // work a little better
      bool flipY = false;
      if (ix==Nfft/2) {
	if (iy==Nfft/2) {
	  // The corner point at ix=iy=Nfft/2
	  // is also self-conjugate, extra factor 1/2
	  flipY = true;
	  z *= 0.25;
	} else if (iy>0) {
	  flipY = true;
	  z *= 0.5;
	}
      } else if (iy==Nfft/2 && ix>0) {
	flipY = true;
	z *= 0.5;
      }
     
      re.push_back(z.real());
      im.push_back(z.imag());
      psfre.push_back(mtf.real());
      psfim.push_back(mtf.imag());
      vkx.push_back(kx);
      vky.push_back(ky);

      if (flipY) {
	re.push_back(z.real());
	im.push_back(z.imag());
	psfre.push_back(mtf.real());
	psfim.push_back(mtf.imag());
	vkx.push_back(kx);
	vky.push_back(-ky);
      }
    } // ix loop
  } // iy loop
  } // close critical block  
  /**cerr << "SbGalaxy dk " << dk
	   << " Nfft " << Nfft
	   << " kmaxsq " << kmaxsq
	   << " kept " << re.size()
	   << endl;
  /**/
  KData kgal;
  kgal.set(re, im, vkx, vky, d2k, dcIndex);

  KData kpsf;
  kpsf.set(psfre, psfim, vkx, vky, d2k, dcIndex);

  return new PixelGalaxy<UseMoments>(kw, kgal, kpsf, noise);
}

///////////////////////////////////////////////////////////////
// 
// Instantiations of the templates
//
///////////////////////////////////////////////////////////////


#define INSTANTIATE(u) \
  template PixelGalaxy<u>* bfd::SbGalaxy<u>(const KWeight&, const sbp::SBProfile&, \
				       const Psf&, ran::UniformDeviate&, double, bool);

#include "InstantiateMomentCases.h"

