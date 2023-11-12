// Classes to build Galaxy from pixel data
#ifndef PIXELGALAXY_H
#define PIXELGALAXY_H
 
#include "Galaxy.h"
#include "Image.h"


namespace bfd {

  // Interface to a PSF.  We are going to assume that PSF is given in SKY units
  // and should =1 at k=0 if it conserves flux.
  class Psf {
  public:
    virtual ~Psf() {}

    virtual DComplex kValue(double kx, double ky) const =0;
    virtual CVector kValue(const DVector& kx, const DVector& ky) const {
      Assert(kx.size()==ky.size());
      CVector out(kx.size());
      for (int i=0; i<kx.size(); i++) out[i] = kValue(kx[i],ky[i]);
      return out;
    }
    virtual double kMax() const=0;

  };

  // Delta-function PSF.  Return some large value for kMax.
  class DeltaPsf: public Psf {
  public:
    virtual DComplex kValue(double kx, double ky) const {return DComplex(1.,0.);}
    virtual double kMax() const {return 1e9;}
  };
    
  // Class that holds value of a Hermitian complex function over a grid of points in k space.
  class KData {
  public:
    DVector re;
    DVector im;
    DVector kx;
    DVector ky;
    double  d2k;	// This is the area of a grid cell in k space - turns sums over (kx,ky) into integrals
    int     dcIndex;	// vector index of the point at zero frequency, which has no conjugate.

    KData(int size=0): re(size,0.), im(size,0.), kx(size,0.), ky(size,0.), d2k(0.), dcIndex(-1) {}
    // Fill with any standard-library container type
    template <class C>
    void set(const C& vre, const C& vim, const C& vkx, const C& vky, 
	     double d2k_, int dc) {
      resize(vre.size());
      d2k = d2k_;
      dcIndex = dc;
      std::copy(vre.begin(), vre.end(), re.begin());
      Assert(vim.size()==im.size()); std::copy(vim.begin(), vim.end(), im.begin());
      Assert(vkx.size()==kx.size()); std::copy(vkx.begin(), vkx.end(), kx.begin());
      Assert(vky.size()==ky.size()); std::copy(vky.begin(), vky.end(), ky.begin());
    }

    void resize(int size) {
      re.resize(size);
      im.resize(size);
      kx.resize(size);
      ky.resize(size);
    }

    // Apply phase change to move coord origin by (dx,dy) or move object by (-dx,-dy):
    KData getShifted(double dx, double dy);
  };

  // Class that represents an affine transformation in 2d.
  // Forward transformation is x_out = a*(x_in - x0).
  class Affine {
  public:
    // Default is the identity transformation
    Affine(): x0(0.) {a.setToIdentity(); ainv.setToIdentity(); ainvT.setToIdentity();}
    // If just a center is given, matrix is set to identity
    Affine(const DVector2& x0_): x0(x0_) {
      a.setToIdentity(); 
      ainv.setToIdentity();
      ainvT.setToIdentity();}
    // Can give the full matrix
    Affine(const DMatrix22& a_, const DVector2& x0_): a(a_), x0(x0_) {
      ainv=a.inverse();
      ainvT = ainv.transpose();
    }
    // This constructor assumes map is just a scaling plus optional translation
    Affine(double dx, const DVector2& x0_=DVector2(0.)): x0(x0_) {
      a.setToIdentity();
      a *= dx;
      ainv = a.inverse();
      ainvT = ainv.transpose();
    }

    // Apply the transformation:
    DVector2 fwd(const DVector2& x_in) const {return a*(x_in-x0);}
    DVector2 inv(const DVector2& x_in) const {return ainv*x_in + x0;}
    // Give k value in out units that is same wave as k_in in input units (k.x is conserved)
    DVector2 kFwd(const DVector2& k_in) const {return ainvT * k_in;}

    double getX0() const {return x0[0];}
    double getY0() const {return x0[1];}
    // ?? bulk transformations ??

    // Jacobian determinant of the forward transformation:
    double jacobian() const {return a(0,0)*a(1,1)-a(0,1)*a(1,0);}
  private:
    DMatrix22 a;
    DVector2 x0;
    DMatrix22 ainv;
    DMatrix22 ainvT;	// Transpose of ainv
  };

  template<class CONFIG>
  class WKMoments;

  template<class CONFIG>
  class WKDerivs {
    friend class WKMoments<CONFIG>;  // So we can slice off just the moments columns
  public:
    // Constructor will build the cofactors for integration with the 
    // given weight function and the (kx,ky) samples in KData.
    WKDerivs(const KWeight& kw, const KData& kd);
    // Return a moments/derivatives decomposition by integrating with the image in KData:
    TemplateGalaxy<CONFIG> getTemplate(const KData& kd) const;

  private:
    DMatrix evens;
    DMatrix odds;
  };

  template<class CONFIG>
  class WKMoments {
  public:
    // Constructor will build the cofactors for integration with the 
    // given weight function and the (kx,ky) samples in KData.
    WKMoments(const KWeight& kw, const KData& kd);

    // Return a moments vector by integrating with the image in KData:
    Moment<CONFIG> getMoment(const KData& kd) const;

    // Return the moment covariance that an image with unit white noise power
    // will have.  The KData specifies the FT of the PSF on the K grid.
    MomentCovariance<CONFIG> getCov(const KData& psf) const;

  private:
    DMatrix evens;
    DMatrix odds;
  };

  // This is the class that will measure moments from pixelized data (or directly from an array of
  // k-space points).  Note this class saves intermediate products that can take a lot of space:
  // the PSF values at all k points, plus the weights at all k points for moments & their derivs.
  // So the PixelGalaxy objects should be deleted after you have gotten the needed moments.

  // If you know that the k array and weight function will be the same for all galaxies, you
  // can extract the weight matrices (WKMoments or WKDerivs) from one PixelGalaxy and then have
  // all new ones copy and use it instead of recalculating.

  template<class CONFIG>
  class PixelGalaxy: public GalaxyData<CONFIG>
    {
    public:
      typedef CONFIG BC;
      typedef typename CONFIG::FP FP;

      // Construct the galaxy moments from image data and an affine map from pixel to object coords.
      // FFT will be sized to the image, with zero padding to get a good FFT size.
      // The image should be given in units of flux per pixel area.
      // The noise parameter is the white-noise level of the image (variance per sqrt(pixel)), i.e.
      // it is the counts per pixel if this is a bgrnd-limited image in electron units.
      PixelGalaxy(const KWeight& kw_, 
		  const img::Image<> img,
		  const Affine& map,
		  const Psf& psf, 
		  double noise_=1.);

      // Construct the galaxy moments using already-deconvolved k-space galaxy data
      // and already-looked up psf data.  ASSUMES that the k values match up.
      // The noise parameter in this case should be the white-noise level in whatever coordinate
      // system the k values are using.
      PixelGalaxy(const KWeight& kw_, 
		  const KData& gal,
		  const KData& psf, 
		  double noise_=1.);

      void operator=(const PixelGalaxy& rhs) =delete;
      virtual ~PixelGalaxy();
      PixelGalaxy(const PixelGalaxy& rhs);

      virtual TargetGalaxy<CONFIG> getTarget(bool fillCovariance=true) const;
      virtual TemplateGalaxy<CONFIG> getTemplate() const;
      virtual PixelGalaxy* getShifted(double dx, double dy) const;

      const WKMoments<CONFIG>* getWKMoments() const;  // Calculate the cofactors to KData for moments
      void copyWKMoments(const WKMoments<CONFIG>& wkmIn) const; // Adopt these precomputed cofactors

      // Calculate the cofactors to KData for moments decomposition
      const WKDerivs<CONFIG>* getWKDerivs() const;
      void copyWKDerivs(const WKDerivs<CONFIG>& wkdIn) const; // Adopt these precomputed cofactors

    private:
      const KWeight& kw;
      KData   kd;
      KData   psfData;

      typename MI::Type noise;

      // cache the integration coefficients for moments / derivs
      mutable WKMoments<CONFIG>* wkm;
      mutable WKDerivs<CONFIG>*  wkd;

};

} // namespace bfd

#endif

