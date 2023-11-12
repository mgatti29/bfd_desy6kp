// Galaxy defined by samples of k space
#ifndef KGALAXY_H
#define KGALAXY_H
 
#include "BfdConfig.h"
#include "Galaxy.h"
#include <set>
#include <list>

namespace bfd {

  // KWeight Base Class - template is float or double precision

  template <class FP>
  class KWeight
  // KWeight is a circularly symmetric function in 2d k space.
  {
  public:
    typedef linalg::Vector<FP> Vector;
    typedef linalg::Matrix<FP> Matrix;
    const int W=0;  // index of the weight itself
    const int WP=1;  // index of the 1st deriv of weight wrt k^2
    const int WPP=2;  // index of the 2nd deriv of weight wrt k^2
    
    // Return the weight function, given the argument |k^2|
    virtual FP operator()(FP ksq) const =0;
    // Return the weight function and its first two derivatives with respect to the
    // argument |k^2|:
    virtual void derivatives(FP& w, FP& dw_dksq, FP& d2w_dksq2, 
			     FP ksq) const =0;
    // Return the k value beyond which the weight is zero or negligible:
    virtual FP kMax() const=0;

    // Array operation forms:
    virtual Vector operator()(const Vector& ksq) const {
      Vector out(ksq.size());
      for (int i=0; i<ksq.size(); i++) out[i] = this->operator()(ksq[i]);
      return out;
    }
    virtual Matrix derivatives(const Vector& ksq) const {
      Matrix out(ksq.size(),3);
      FP w, wp, wpp;
      for (int i=0; i<ksq.size(); i++) {
	derivatives(w, wp, wpp,ksq[i]);
	out(i,W) = w; out(i,WP) = wp; out(i,WPP) = wpp;
      }
      return out;
    }
  };

  // Gaussian Fourier-domain Weight Class
  // Need to specify just the width.
  // Note that wsigma_x input is the x-domain width.
  template <typename FP>
  class GaussianWeight: public KWeight<FP>
  {
  public:
    GaussianWeight(FP wsigma_x): sigxsq(wsigma_x*wsigma_x) {}
    virtual FP operator()(FP ksq) const {
      return exp(FP(-0.5)*ksq*sigxsq);
    }
    virtual void derivatives(FP& w, FP& dw_dksq, FP& d2w_dksq2, 
			     FP ksq) const {
      w = (*this)(ksq);
      dw_dksq = FP(-0.5) * sigxsq * w;
      d2w_dksq2 = FP(-0.5) * sigxsq * dw_dksq ;
    }
    virtual FP kMax() const {return FP(5.)/getSigma();} // This is approximate.

    // Accessor function to sigma 
    FP getSigma() const {return sqrt(sigxsq);}

  private:
    FP sigxsq;
  };

  // Radial weight of (1-(k*s)**2)**n
  // It will force n>=2
  // The argument sigma is roughly equivalent to Gaussian sigma, 2*n*s^2 = sigma^2
  // Internal variable ssq is the inverse square of cutoff frequency.
  template <typename FP>
  class KSigmaWeight: public KWeight<FP>
  {
  public:
    KSigmaWeight(FP sigma, int n=4): ssq(sigma*sigma/(2*n)), nMinus2(n-2) {
      if (nMinus2 < 0) nMinus2 = 0;
    }
    virtual FP kMax() const {return 1./sqrt(ssq);}
    virtual FP operator()(FP ksq) const {
      FP u = FP(1.) - ksq *ssq;
      if (u<=0.) return 0.;
      FP out = u*u;
      for (int i=0; i<nMinus2; i++) out *= u;
      return out;
    }
    virtual void derivatives(FP& w, FP& dw_dksq, FP& d2w_dksq2, 
			     FP ksq) const {
      FP u = FP(1.) - ksq *ssq;
      if (u<=0.) {
	w = dw_dksq = d2w_dksq2 = 0.;
      } else {
	FP out = 1.;
	for (int i=0; i<nMinus2; i++) out *= u;
	w = out * u * u;
	dw_dksq = -out * u * ssq * (nMinus2+2);
	d2w_dksq2 = out * ssq * ssq * (nMinus2+2) * (nMinus2+1);
      }
    }
  private:
    FP ssq;
    int nMinus2;
  };
    
  template <class CONFIG>
  class KGalaxy {
  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    typedef linalg::Vector<FP> Vector;
    typedef linalg::Vector<std::complex<FP>> CVector;
    typedef linalg::Matrix<FP> Matrix;
    typedef linalg::Matrix<std::complex<FP>> CMatrix;
    typedef TemplateGalaxy<BC> TG;
    typedef std::complex<double> DComplex;

    // Construct with or without info on variance:
    KGalaxy(const KWeight<FP>& kw,		// K-space weight function to use
	    const CVector& kval_,	// FT values at each sample
	    const Vector& kx,		// kx and ky values at each sample.
	    const Vector& ky,
	    FP d2k,		// Area in k space assigned to each sample
	    // Indices of k's that do *not* also represent conjugate
	    const std::set<int>& unconjugated,
	    // Galaxy sky position:
	    const linalg::DVector2& posn_=linalg::DVector2(0.),
	    const long id_=0L);  
    KGalaxy(const KWeight<FP>& kw,
	    const CVector& kval_,
	    const Vector& kx,
	    const Vector& ky,
	    const Vector& kvar_,	// Variance at each k, more precisely cov(kval, kval.conjugate())
	    FP d2k, 
	    const std::set<int>& unconjugated,
	    // Galaxy sky position:
	    const linalg::DVector2& posn_=linalg::DVector2(0.),
	    const long id_=0L);  

    virtual ~KGalaxy() {
      delete kvar;  //ok to delete nullptr
    }

    KGalaxy(const KGalaxy& rhs);
    KGalaxy(KGalaxy&& rhs); // Move constructor adopts arrays
    KGalaxy& operator=(const KGalaxy& rhs);
    KGalaxy& operator=(KGalaxy&& rhs);
    
    // Create a TargetGalaxy from the data, with or w/o covariance
    virtual TargetGalaxy<CONFIG> getTarget(bool fillCovariance=true) const;
    // Create a TemplateGalaxy from the data.
    virtual TemplateGalaxy<CONFIG> getTemplate() const;
    // Return new data with coordinate origin moved by (dx,dy)
    virtual KGalaxy* getShifted(double dx, double dy) const;

    EIGEN_NEW
  private:
    CVector kval;  // The FFT values, PSF-corrected, with area factors
    CVector kz;    // The k values at each point, kx + i*ky
    Vector* kvar;  // Cov(kval, kval.conjugate()) at each k, if given, with dk's.
    const KWeight<FP> *wt;  // The weight function being used
    linalg::DVector2 posn;  // Galaxy position
    long id;                // Galaxy ID number

    // These classes are used to construct the moment derivatives
    // from multipole moments of the galaxy:
    struct TemplateTerm {
      int mIndex;	// Which moment it contributes to
      int dIndex;   // Which derivative it contributes to
      DComplex coeff;  // Coefficient of the term
      int m;        // Multipole number (p-q)
      int N;        // total power of k (p+q)
    TemplateTerm( int mIndex_, int dIndex_, DComplex coeff_, int m_, int N_):
      mIndex(mIndex_),
	dIndex(dIndex_),
	coeff(coeff_),
	m(m_),
	N(N_) {}
    };
    static list<TemplateTerm> w0TermsEven;
    static list<TemplateTerm> w0TermsOdd;
    static list<TemplateTerm> w1TermsEven;
    static list<TemplateTerm> w1TermsOdd;
    static list<TemplateTerm> w2TermsEven;
    static list<TemplateTerm> w2TermsOdd;
    void addTerms(const std::list<TemplateTerm>& terms,
		  typename TG::DerivMatrix& destination,
		  const Vector& vw,
		  bool isFalse) const;
    
  };

} // namespace bfd

#endif   // KGALAXY_H
