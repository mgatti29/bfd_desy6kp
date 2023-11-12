// Class that will give Gaussian-weighted k moments of a Gaussian 
#ifndef GAUSS_MOMENTS_H
#define GAUSS_MOMENTS_H
#include "Std.h"
#include "LinearAlgebra.h"

namespace bfd {

  class GaussMoments {
  private:
    double flux;
    double sigmax;
    double sigmay;
    double sigsq;
    double sigsqW;
    double e;
    double beta;
    double x0;
    double y0;

    // Other derived quantities:
    DComplex u;  // (x0/sigx^2,y0/sigy^2) in diagonal frame.
    mutable linalg::Matrix<double> Y;  // Moments at x0=y0=beta=0.
    double x0Suppression; // Exponential factor of centroid that appears in all moments.

    // Calculate the moments of k^p kbar^q for a centered, unrotated Gaussian:
    void setY() const;
    // Calculate the u value and x0 suppression factor
    void setU();

    static const int pqMax=8;  // Highest value of p + q needed in moments

  public:
    // Note that the sigma given to this program is such that 2*sigma^2 is the TRACE
    // of the COVARIANCE matrix of the Gaussian.
    GaussMoments(double flux_, double sigma, double e_,
		 double sigmaW, 
		 double beta_ = 0.,
		 double x0_ = 0.,
		 double y0_ = 0.): e(e_), beta(beta_), flux(flux_), Y(pqMax+1,pqMax+1) {
      sigsq = sigma*sigma;
      sigsqW = sigmaW * sigmaW;
      sigmax = sqrt( sigsq*(1+e) + sigsqW);
      sigmay = sqrt( sigsq*(1-e) + sigsqW);
      setY();
      setX0(x0_, y0_);
    }
    // Default copy and destructor are fine.

    void setX0(double x0_, double y0_) ;
    void setX0(DComplex x0_) {setX0(real(x0_), imag(x0_));}
    void setBeta(double beta_);
    DComplex getX0() const {return DComplex(x0,y0);}
    double getBeta() const {return beta;}
    double getE() const {return e;}

    // Return the moment of k^p kbar^q for current center and rotation
    DComplex moment(int p, int q) const;
  };

} // end namespace bfd
#endif // GAUSS_MOMENTS_H
