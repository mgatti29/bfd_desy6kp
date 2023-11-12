#ifndef KWEIGHT_H
#define KWEIGHT_H

# include <iostream>

namespace bfd
{
  // KWeight Base Class

  class KWeight
  // KWeight is a circularly symmetric function in 2d k space.
  {
  public:
    // Return the weight function, given the argument |k^2|
    virtual double operator()(double ksq) const =0;
    // Return the weight function and its first two derivatives with respect to the
    // argument |k^2|:
    virtual void derivatives(double& w, double& dw_dksq, double& d2w_dksq2, 
			     double ksq) const =0;
    // Return the k value beyond which the weight is zero or negligible:
    virtual double kMax() const=0;

    // Array operation forms:
    virtual DVector operator()(const DVector& ksq) const {
      DVector out(ksq.size());
      for (int i=0; i<ksq.size(); i++) out[i] = this->operator()(ksq[i]);
      return out;
    }
    virtual DMatrix derivatives(const DVector& ksq) const {
      DMatrix out(ksq.size(),3);
      double w, wp, wpp;
      for (int i=0; i<ksq.size(); i++) {
	derivatives(w, wp, wpp,ksq[i]);
	out(i,0) = w; out(i,1) = wp; out(i,2) = wpp;
      }
      return out;
    }
  };

  // Gaussian Fourier-domain Weight Class
  // Need to specify just the width.  Note that wsigma_x input is the x-domain width.
  class GaussianWeight: public KWeight
  {
  public:
    GaussianWeight(double wsigma_x): sigxsq(wsigma_x*wsigma_x) {}
    virtual double operator()(double ksq) const {
      return exp(-0.5*ksq*sigxsq);
    }
    virtual void derivatives(double& w, double& dw_dksq, double& d2w_dksq2, 
			     double ksq) const {
      w = (*this)(ksq);
      dw_dksq = -0.5 * sigxsq * w;
      d2w_dksq2 = -0.5 * sigxsq * dw_dksq ;
    }
    virtual double kMax() const {return 5./getSigma();} // This is approximate.

    // Accessor function to sigma 
    double getSigma() const {return sqrt(sigxsq);}

  private:
    double sigxsq;
  };

  // Radial weight of (1-(k*s)**2)**n
  // It will force n>=2
  // ??? Might want to template this with integer n as template
  // The argument sigma is roughly equivalent to Gaussian sigma, 2*n*s^2 = sigma^2
  // Internal variable ssq is the inverse square of cutoff frequency.
  class KSigmaWeight: public KWeight
  {
  public:
    KSigmaWeight(double sigma, int n=4): ssq(sigma*sigma/(2*n)), nMinus2(n-2) {
      if (nMinus2 < 0) nMinus2 = 0;
    }
    virtual double kMax() const {return 1./sqrt(ssq);}
    virtual double operator()(double ksq) const {
      double u = 1. - ksq *ssq;
      if (u<=0.) return 0.;
      double out = u*u;
      for (int i=0; i<nMinus2; i++) out *= u;
      return out;
    }
    virtual void derivatives(double& w, double& dw_dksq, double& d2w_dksq2, 
			     double ksq) const {
      double u = 1. - ksq *ssq;
      if (u<=0.) {
	w = dw_dksq = d2w_dksq2 = 0.;
      } else {
	double out = 1.;
	for (int i=0; i<nMinus2; i++) out *= u;
	w = out * u * u;
	dw_dksq = -out * u * ssq * (nMinus2+2);
	d2w_dksq2 = out * ssq * ssq * (nMinus2+2) * (nMinus2+1);
      }
    }
  private:
    double ssq;
    int nMinus2;
  };
} // end namespace bfd

#endif // KWEIGHT_H
