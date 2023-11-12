#ifndef GAUSSIANGALAXY_H
#define GAUSSIANGALAXY_H

#include "BfdConfig.h"
#include "Moment.h"
#include "Galaxy.h"
#include "GaussMoments.h"
 
namespace bfd
{

  // Gaussian Galaxy class gives all moments etc for Gaussian galaxies
  // with Gaussian weight function, done analytically.
  // The sigma used by GaussianGalaxy is the SQRT of half of the TRACE of the COVARIANCE matrix,
  // 2*sigma^2 = Tr(Cov)
  // Which is sqrt( det(A)/sqrt(1-e^2)) where A is coord transformation matrix.
  template<class CONFIG>
  class GaussianGalaxy: public GalaxyData<CONFIG> {
  public:
    typedef CONFIG BC;
    typedef typename CONFIG::FP FP;
    GaussianGalaxy(double flux_, 
		   double sigma_, double e_, double beta_,
		   double cx_, double cy_,
		   double wtSigma_=1.,
		   double noise_=1.): flux(flux_),
      noise(noise_),
      sigma(sigma_),
      e(e_),
      beta(beta_),
      cx(cx_),
      cy(cy_),
      wtSigma(wtSigma_) {}

    virtual ~GaussianGalaxy() {}
    
    void dump(ostream& oss) const {
      // dump galaxy characteristics to the stream
      oss << "Flux " << flux
	  << " e " << e
	  << " beta " << beta*180./PI
	  << " sigma " << sigma
	  << " S/N " << flux / sqrt(4*PI*sigma*sigma*noise)
	  << " center " << cx << "," << cy
	  << " wtSigma " << wtSigma;
    }

    virtual TargetGalaxy<CONFIG> getTarget(bool fillCovariance=true) const {
      GaussMoments all_moments(flux, sigma, e, wtSigma, beta, cx, cy);
    
      // First fill a moment structure
      Moment<BC> mom;
      mom.m[BC::MF] = real(all_moments.moment(0,0));
      mom.m[BC::MR] = real(all_moments.moment(1,1));
      mom.m[BC::M1] = real(all_moments.moment(2,0));
      mom.m[BC::M2] = imag(all_moments.moment(2,0));
      if (BC::UseConc)
	mom.m[BC::MC] = real(all_moments.moment(2,2));
      
      if (!BC::FixCenter) {
	mom.xy[BC::MX] = -imag(all_moments.moment(1,0));
	mom.xy[BC::MY] = real(all_moments.moment(1,0));
      }

      // Then, if desired, covariance
      MomentCovariance<CONFIG> cov;
      if (fillCovariance) {
	FP invsigsq = 1./(wtSigma*wtSigma);
	FP factor = 4*PI*PI*PI * noise * invsigsq;

	cov.m.setZero();
	cov.m(BC::MF,BC::MF) = factor;
	cov.m(BC::MR,BC::MF) = factor * invsigsq;
	cov.m(BC::MF,BC::MR) = factor * invsigsq;
	cov.m(BC::MR,BC::MR) = factor * FP(2.) * invsigsq * invsigsq;
	cov.m(BC::M1,BC::M1) = factor * invsigsq*invsigsq;
	cov.m(BC::M2,BC::M2) = factor * invsigsq*invsigsq;
	if (BC::UseConc) {
	  cov.m(BC::MC,BC::MF) = factor * invsigsq * invsigsq * FP(2.);
	  cov.m(BC::MF,BC::MC) = factor * invsigsq * invsigsq * FP(2.);
	  cov.m(BC::MC,BC::MR) = factor * invsigsq * invsigsq * invsigsq * FP(6.);
	  cov.m(BC::MR,BC::MC) = factor * invsigsq * invsigsq * invsigsq * FP(6.);
	  cov.m(BC::MC,BC::MC) = factor * invsigsq * invsigsq * invsigsq * invsigsq * FP(24.);
	}
	
	if (!BC::FixCenter) {
	  cov.xy.setZero();
	  cov.xy(BC::MX,BC::MX) = factor * FP(0.5) * invsigsq;
	  cov.xy(BC::MY,BC::MY) = factor * FP(0.5) * invsigsq;
	}
      }

      linalg::DVector2 xy;
      xy[0] = cx; xy[1] = cy;
      return TargetGalaxy<CONFIG>(mom, cov, xy);
    }

    virtual TemplateGalaxy<CONFIG> getTemplate() const {
      typedef TemplateGalaxy<CONFIG> TG;
      // Do complex arithmetic in double precision since GaussMoments does.
      const std::complex<double> I(FP(0.),FP(1.));
      // Factors appearing for first and second derivatives of weight function:
      double d1 = -0.5 * wtSigma * wtSigma;
      double d2 = d1 * d1;
  
      TG out;
      
      GaussMoments gm(flux, sigma, e, wtSigma, beta, cx, cy);

      // Fill in moments and derivs for even moments
      out.mDeriv(TG::MF, TG::D0) = gm.moment(0,0);
      out.mDeriv(TG::MR, TG::D0) = gm.moment(1,1);
      out.mDeriv(TG::ME, TG::D0) = gm.moment(2,0);

      out.mDeriv(TG::MF, TG::DV) = -d1*gm.moment(0,2);
      out.mDeriv(TG::MR, TG::DV) = -gm.moment(0,2) -d1*gm.moment(1,3);
      out.mDeriv(TG::ME, TG::DV) = -2.*gm.moment(1,1) -d1*gm.moment(2,2);

      out.mDeriv(TG::MF, TG::DVb) = -d1*gm.moment(2,0);
      out.mDeriv(TG::MR, TG::DVb) = -gm.moment(2,0) -d1*gm.moment(3,1);
      out.mDeriv(TG::ME, TG::DVb) = -d1*gm.moment(4,0);

      out.mDeriv(TG::MF, TG::DV_DV) = d2 * gm.moment(0,4);
      out.mDeriv(TG::MR, TG::DV_DV) = 2.*d1*gm.moment(0,4) + d2*gm.moment(1,5);
      out.mDeriv(TG::ME, TG::DV_DV) = 2.*gm.moment(0,2) +4.*d1*gm.moment(1,3) + d2*gm.moment(2,4);

      out.mDeriv(TG::MF, TG::DVb_DVb) = d2*gm.moment(4,0);
      out.mDeriv(TG::MR, TG::DVb_DVb) = 2.*d1*gm.moment(4,0) + d2*gm.moment(5,1);
      out.mDeriv(TG::ME, TG::DVb_DVb) = d2*gm.moment(6,0);

      out.mDeriv(TG::MF, TG::DV_DVb) = 2.*d1*gm.moment(1,1) + d2*gm.moment(2,2);
      out.mDeriv(TG::MR, TG::DV_DVb) = 2.*gm.moment(1,1)  +4.*d1*gm.moment(2,2) + d2*gm.moment(3,3);
      out.mDeriv(TG::ME, TG::DV_DVb) = gm.moment(2,0) +4.*d1*gm.moment(3,1) + d2*gm.moment(4,2);

      if (BC::UseMag) {
	// Magnification derivatives too:
	out.mDeriv(TG::MF, TG::DU) = -2.*d1 * gm.moment(1,1);
	out.mDeriv(TG::MR, TG::DU) = -2.*gm.moment(1,1) - 2.*d1*gm.moment(2,2);
	out.mDeriv(TG::ME, TG::DU) = -2.*gm.moment(2,0) -2.*d1*gm.moment(3,1);
	
	out.mDeriv(TG::MF, TG::DU_DU) = 6.*d1 * gm.moment(1,1) + 4.*d2*gm.moment(2,2);
	out.mDeriv(TG::MR, TG::DU_DU) = 6.*gm.moment(1,1) +14.*d1*gm.moment(2,2) + 4.*d2*gm.moment(3,3);
	out.mDeriv(TG::ME, TG::DU_DU) = 6.*gm.moment(2,0) +14.*d1*gm.moment(3,1) + 4.*d2*gm.moment(4,2);

	out.mDeriv(TG::MF, TG::DU_DV) = 2.*d1 * gm.moment(0,2) + 2.*d2*gm.moment(1,3);
	out.mDeriv(TG::MR, TG::DU_DV) = 2.*gm.moment(0,2) +6.*d1*gm.moment(1,3) + 2.*d2*gm.moment(2,4);
	out.mDeriv(TG::ME, TG::DU_DV) = 4.*gm.moment(1,1) +8.*d1*gm.moment(2,2) + 2.*d2*gm.moment(3,3);

	out.mDeriv(TG::MF, TG::DU_DVb) = 2.*d1 * gm.moment(2,0) + 2.*d2*gm.moment(3,1);
	out.mDeriv(TG::MR, TG::DU_DVb) = 2.*gm.moment(2,0) +6.*d1*gm.moment(3,1) + 2.*d2*gm.moment(4,2);
	out.mDeriv(TG::ME, TG::DU_DVb) = 4.*d1*gm.moment(4,0) + 2.*d2*gm.moment(5,1);
      }
      
      if (BC::UseConc) {
	out.mDeriv(TG::MC, TG::D0) = gm.moment(2,2);
	out.mDeriv(TG::MC, TG::DV) = -2.*gm.moment(1,3) - d1*gm.moment(2,4);
	out.mDeriv(TG::MC, TG::DVb) = -2.*gm.moment(3,1) - d1*gm.moment(4,2);
	out.mDeriv(TG::MC, TG::DV_DV) = 2.*gm.moment(0,4) + 4.*d1*gm.moment(1,5) +d2*gm.moment(2,6);
	out.mDeriv(TG::MC, TG::DVb_DVb) = 2.*gm.moment(4,0) + 4.*d1*gm.moment(5,1) +d2*gm.moment(6,2);
	out.mDeriv(TG::MC, TG::DV_DVb) = 6.*gm.moment(2,2) + 6.*d1*gm.moment(3,3) + d2*gm.moment(4,4);

	if (BC::UseMag) {
	  out.mDeriv(TG::MC, TG::DU) = -4.*gm.moment(2,2) -2.*d1*gm.moment(3,3);
	  out.mDeriv(TG::MC, TG::DU_DU) = 20.*gm.moment(2,2) + 22.*d1*gm.moment(3,3) + 4.*d2*gm.moment(4,4);
	  out.mDeriv(TG::MC, TG::DU_DV) = 8.*gm.moment(1,3) + 10.*d1*gm.moment(2,4) + 2.*d2*gm.moment(3,5);
	  out.mDeriv(TG::MC, TG::DU_DVb) = 8.*gm.moment(3,1) + 10.*d1*gm.moment(4,2) + 2.*d2*gm.moment(5,3);
	}
      }

      if (!BC::FixCenter) {
	out.xyDeriv(TG::MX,TG::D0) = I * gm.moment(1,0);
	out.xyDeriv(TG::MX,TG::DV) = -I * gm.moment(0,1) - I*d1*gm.moment(1,2);
	out.xyDeriv(TG::MX,TG::DVb) = -I * d1*gm.moment(3,0);
	out.xyDeriv(TG::MX,TG::DV_DV) = 2.*I * d1*gm.moment(0,3) + I*d2*gm.moment(1,4);
	out.xyDeriv(TG::MX,TG::DVb_DVb) = I*d2*gm.moment(5,0);
	out.xyDeriv(TG::MX,TG::DV_DVb) = 0.5* I * gm.moment(1,0) + 3.*I*d1*gm.moment(2,1) + I*d2*gm.moment(3,2);
	if (BC::UseMag) {
	  out.xyDeriv(TG::MX,TG::DU) = -I * gm.moment(1,0) -2.*I*d1*gm.moment(2,1);
	  out.xyDeriv(TG::MX,TG::DU_DU) = 2.*I * gm.moment(1,0) +10.*I*d1*gm.moment(2,1) +4.*I*d2*gm.moment(3,2);
	  out.xyDeriv(TG::MX,TG::DU_DV) = I * gm.moment(0,1) + 5.*I*d1*gm.moment(1,2) + 2.*I*d2*gm.moment(2,3);
	  out.xyDeriv(TG::MX,TG::DU_DVb) = 3.*I*d1*gm.moment(3,0) + 2.*I*d2*gm.moment(4,1);
	}
      }
      return out;
    }

    // Return another Galaxy which is this one with coordinate origin shifted by (dx,dy)
    // i.e. object moves by (-dx, -dy).
    virtual GaussianGalaxy<CONFIG>* getShifted(double dx, double dy) const {
      auto copy = new GaussianGalaxy<CONFIG>(*this);
      copy->cx-=dx;
      copy->cy-=dy;
      return copy;
    }

  private:
    // Total flux:
    double flux;
    // Shape:
    double sigma;
    double beta;
    double e;
    // Center coordinates:
    double cx;
    double cy;
    // Image white noise level:
    double noise;
    // Weight function sigma
    double wtSigma;
  };


} // end namespace bfd

#endif // GAUSSIANGALAXY_H
