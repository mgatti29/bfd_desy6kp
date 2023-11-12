// Test the analytic GaussianGalaxy formulae.
// First test moments and covariance against numerical instances.
// Then test derivatives against finite differences of analytic moments.

// ??? clear up sigma value in Ellipse, numerical moments work with pow() shut off.
// ??? Mu derivatives v bad... another sigma thing?
#include <algorithm>
#include "Shear.h"
#include "BfdConfig.h"
#include "GaussianGalaxy.h"

using namespace bfd;
// Test case will exercise all moments and derivs in double precision.
const bool FIX_CENTER=false;
const bool USE_CONC = true;
const bool USE_MAG = true;
const int N_COLORS=0;
const bool USE_FLOAT=false;
typedef BfdConfig<FIX_CENTER,
		  USE_CONC,
		  USE_MAG,
		  N_COLORS,
		  USE_FLOAT> BC;

typedef BC::FP FP;
using linalg::DVector;
using linalg::CVector;

GaussianGalaxy<BC> 
ellipse2galaxy(double wtSigma, double flux, const Ellipse& el, double noise) {
  double e =  el.getS().getE();
  double beta = el.getS().getBeta();
  Position<double> c = el.getX0();
  // The GaussianGalaxy wants sigma^2 to be the trace of the covariance matrix.
  double sigma = exp(el.getMu()) * pow(1-e*e, -0.25);
  return GaussianGalaxy<BC>(flux, sigma, e, beta, c.x, c.y, wtSigma, noise);
}

// Function to extract one derivative from a template in Moment form
Moment<BC> derivOf(const TemplateGalaxy<BC>& tmpl, int i) {
  typename BC::MVector dm(tmpl.realMDerivs().col(i));
  if (BC::FixCenter)
    return Moment<BC>(dm);
  else {
    typename BC::XYVector dxy(tmpl.realXYDerivs().col(i));
    return Moment<BC>(dm,dxy);
  }
}

bool
compare(const DVector& analytic, const DVector& numeric) {
  const double TOLERANCE = 1e-4;
  const double FTOL = 1e-3;
  int N = analytic.size();
  bool failure = false;
  for (int i=0; i<N; i++) {
    double diff = analytic[i] - numeric[i];
    if (abs(diff) > max(TOLERANCE, FTOL*abs(analytic[i]))) {
      failure = true;
    }
  }
  if (failure) cout << "========>FAILURE:" << endl;
  cout << "Analytic: ";
  for (int i=0; i<N; i++) cout << fixed << setprecision(6) << analytic[i] << " ";
  cout << endl;
  cout << "Numeric:  ";
  for (int i=0; i<N; i++) cout << fixed << setprecision(6) << numeric[i] << " ";
  cout << endl;
  return failure;
}

const double TOLERANCE = 1e-4;
const double FTOL = 1e-3;
bool
compareMoments(const Moment<BC>& m1, const Moment<BC>& m2,
	       string label1="Analytic",string label2="Numeric") {
  bool failure = false;
  for (int i=0; i<BC::MSIZE; i++) {
    double diff = m1.m[i] - m2.m[i];
    if (abs(diff) > max(TOLERANCE, FTOL*abs(m1.m[i]))) {
      failure = true;
    }
  }
  if (!BC::FixCenter) {
    for (int i=0; i<BC::XYSIZE; i++) {
      double diff = m1.xy[i] - m2.xy[i];
      if (abs(diff) > max(TOLERANCE, FTOL*abs(m1.xy[i]))) {
	failure = true;
      }
    }
  }
  int w = std::max(label1.size(), label2.size()) + 2;
  if (failure) cout << "========>FAILURE:" << endl;
  cout << left << setw(w) << label1 + ":";
  for (int i=0; i<BC::MSIZE; i++) cout << fixed << setprecision(6) << m1.m[i] << " ";
  if (!BC::FixCenter) 
    for (int i=0; i<BC::XYSIZE; i++) cout << fixed << setprecision(6) << m1.xy[i] << " ";
  cout << endl;
  cout << left << setw(w) << label2 + ":";
  for (int i=0; i<BC::MSIZE; i++) cout << fixed << setprecision(6) << m2.m[i] << " ";
  if (!BC::FixCenter) 
    for (int i=0; i<BC::XYSIZE; i++) cout << fixed << setprecision(6) << m2.xy[i] << " ";
  cout << endl;
  return failure;
}

bool
compareScalar(double analytic, double numeric) {
  const double TOLERANCE = 1e-4;
  const double FTOL = 1e-3;
  bool failure = false;
  double diff = analytic - numeric;
  if (abs(diff) > max(TOLERANCE, FTOL*abs(analytic))) {
    failure = true;
  }
  if (failure) cout << "FAILURE:" << endl;
  cout << "Analytic: " << fixed << setprecision(6) << analytic
       << " Numeric:  " << fixed << setprecision(6) << numeric
       << endl;
  return failure;
}

int main(int argc,
	 char *argv[])
{
  try {
    double sn = 10.;
    double noise = 1;
    // Intrinsic galaxy shape
    double sigma = 1.5;
    double wtSigma = 1.;
    double e=0.;
    double beta=0.;
    double x0=0.;
    double y0=0.;

    if (argc<2 || argc > 8) {
      x0 = -1.1;
      y0 = 0.3;
      sn = 18.;
      sigma = 2.0;
      e = 0.6;
      wtSigma = 0.5;
      beta = 140. * 3.1415/180.;

      cerr 
	<< "Compare analytic GaussGalaxy moments to numerical.\n"
	"Compare analytic derivatives w.r.t. g1,g2,mu to finite differences.\n"
	"Usage: testGaussShear <sn> [sigma=1.5] [e=0.] [wtSigma=1.] [beta=0.] [x0=0.] [y0=0.]\n"
	" where you give values but will take noted default values starting at rhs if not all args are\n"
	" given.\n"
	" Output is comparison of GaussMoments moment values vs numerical integration.\n"
	" Discrepancies will be marked with ""FAILURE"" \n"
	" You have given no arguments so a standard non-default set is being used.\n"
	" stdout will be comparison of moments.\n"
	" Discrepancies will be marked with ""FAILURE"" and return value !=0 \n"
	" You have given no arguments so a standard non-default set is being used." << endl;
    } 
    if (argc > 1) sn  = atof(argv[1]);
    if (argc > 2) sigma = atof(argv[2]);
    if (argc > 3) e     = atof(argv[3]);
    if (argc > 4) wtSigma= atof(argv[4]);
    if (argc > 5) beta  = atof(argv[5]) * 3.14159 / 180.;
    if (argc > 6) x0    = atof(argv[6]);
    if (argc > 7) y0    = atof(argv[7]);

    // Get the analytic galaxy moments
    double e1 = e * cos(2*beta);
    double e2 = e * sin(2*beta);
    Shear e0(e1, e2);
    Position<double> ctr(x0,y0); 
    double flux = sn * sqrt(4*PI*sigma*sigma*noise);
    Ellipse ell0(e0, log(sigma), ctr);
    GaussianGalaxy<BC> gg0 = ellipse2galaxy(wtSigma, flux, ell0, noise);
    auto targ0 = gg0.getTarget(true);
    auto mom0 = targ0.mom;  // Moment structure

    //*****************************************
    // Check analytic moments vs discrete integration
    //*****************************************

    // Calculate moments from an integration
    double dk = 0.01;
    double kmax = 15.;

    // Build covariance matrix of the Gaussian
    double sigsq = sigma*sigma/sqrt(1-e*e);
    double sigxx = sigsq*(1+e1) + wtSigma*wtSigma;
    double sigyy = sigsq*(1-e1) + wtSigma*wtSigma;
    double sigxy = sigsq*e2;

    bool failure = false;
    typename BC::MVector mNumeric;
    {
      // Even moments first
      CVector sum(BC::MSIZE,0.);
      for (double kx = -kmax; kx<=kmax; kx+=dk) 
	for (double ky = -kmax; ky<=kmax; ky+=dk) {
	  CVector f(BC::MSIZE,0.);
	  f[BC::MF] = 1.;
	  f[BC::MR] = kx*kx+ky*ky;
	  f[BC::M1] = kx*kx-ky*ky;
	  f[BC::M2] = 2. * kx * ky;
	  if (BC::UseConc)
	    f[BC::MC] = f[BC::MR]*f[BC::MR];
	  DComplex z = exp( DComplex(-0.5*(sigxx*kx*kx + sigyy*ky*ky + 2*sigxy*kx*ky),
				     -(kx * x0 + ky*y0)));
	  sum += z*f;
      }
      mNumeric = sum.REAL * flux * dk * dk;
    }

    cout << "Analytic vs Numerical moments:" << endl;
    if (BC::FixCenter) {
      failure = compareMoments(mom0,Moment<BC>(mNumeric));
    } else {
      // Odd moments
      CVector sum(BC::XYSIZE,0.);
      for (double kx = -kmax; kx<=kmax; kx+=dk) 
	for (double ky = -kmax; ky<=kmax; ky+=dk) {
	  CVector f(BC::XYSIZE,0.);
	  f[BC::MX] = DComplex(0,kx);
	  f[BC::MY] = DComplex(0,ky);
	  DComplex z = exp( DComplex(-0.5*(sigxx*kx*kx + sigyy*ky*ky + 2*sigxy*kx*ky),
				     -(kx * x0 + ky*y0)));
	  sum += z*f;
      }
      typename BC::XYVector xyNumeric;
      xyNumeric = sum.REAL * flux * dk * dk;
      failure = compareMoments(mom0, Moment<BC>(mNumeric, xyNumeric)) || failure;
    }
      

    //*****************************************
    // Check analytic derivative vs finite difference
    //*****************************************

    // Now check finite differences of shear and magnification
    const FP dg = 0.002;
    const FP TWO = 2;

    Shear tweak;
    tweak.setG1G2(dg, 0.);
    Ellipse dEl(tweak, 0., Position<double>(0.,0.));
    auto mg1p = ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget(false).mom;
    tweak.setG1G2(-dg, 0.);
    dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
    auto mg1m = ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget(false).mom;
    tweak.setG1G2(0., dg);
    dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
    auto mg2p = ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget(false).mom;
    tweak.setG1G2(0.,-dg);
    dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
    auto mg2m = ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget(false).mom;

    Moment<BC> mmup;
    Moment<BC> mmum;
    if (BC::UseMag) {
      double fac = 1+dg;
      dEl = Ellipse(Shear(), log(fac), Position<double>(0.,0.));
      mmup = ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom;
      fac = 1-dg;
      dEl = Ellipse(Shear(), log(fac), Position<double>(0.,0.));
      mmum = ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom;
    }

    // Get analytic derivatives
    auto tmpl0 = gg0.getTemplate();
    auto mDeriv = tmpl0.realMDerivs();

    // Analytic derivatives of even moments

    // First compare the template's moments to the target's
    cout << "Even moments, Template vs Target: " << endl;
    failure = compareMoments(derivOf(tmpl0, BC::P), mom0, "Template", "Target") || failure;
      
    cout << "G1 derivs: " << endl;
    auto dm = (mg1p-mg1m) / (TWO*dg);
    failure = compareMoments(derivOf(tmpl0, BC::DG1), dm) || failure;

    cout << "G2 derivs: " << endl;
    dm = (mg2p-mg2m) / (TWO*dg);
    failure = compareMoments(derivOf(tmpl0, BC::DG2), dm) || failure;
    
    if (BC::UseMag) {
      cout << "Mu derivs: " << endl;
      dm = (mmup-mmum) / (TWO*dg);
      failure = compareMoments(derivOf(tmpl0, BC::DMU), dm) || failure;
    }
    
    cout << "G1G1 derivs: " << endl;
    dm = (mg1p + mg1m - TWO*mom0) / (dg*dg);
    failure = compareMoments(derivOf(tmpl0, BC::DG1_DG1), dm) || failure;

    cout << "G2G2 derivs: " << endl;
    dm = (mg2p + mg2m - TWO*mom0) / (dg*dg);
    failure = compareMoments(derivOf(tmpl0, BC::DG2_DG2), dm) || failure;
    
    if (BC::UseMag) {
      cout << "MuMu derivs: " << endl;
      dm = (mmup+mmum - TWO*mom0) / (dg*dg);
      failure = compareMoments(derivOf(tmpl0, BC::DMU_DMU), dm) || failure;
    }

    // Crossed 2nd derivatives:
    const FP dg4dg=4.*dg*dg;
    {
      tweak.setG1G2(dg, dg);
      dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
      auto mpp = ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget().mom;
      tweak.setG1G2(dg, -dg);
      dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
      auto mpm = ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget().mom;
      tweak.setG1G2(-dg, dg);
      dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
      auto mmp = ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget().mom;
      tweak.setG1G2(-dg, -dg);
      dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
      auto mmm = ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget().mom;

      cout << "G1G2 derivs: " << endl;
      dm = (mpp + mmm - mpm - mmp) / dg4dg;
      failure = compareMoments(derivOf(tmpl0, BC::DG1_DG2), dm) || failure;
    }

    if (BC::UseMag) {
      double fac = 1+dg;
      tweak.setG1G2(dg, 0);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      auto mpp = ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom;
      tweak.setG1G2(-dg, 0);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      auto mpm = ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom;
      fac = 1-dg;
      tweak.setG1G2(dg, 0);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      auto mmp = ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom;
      tweak.setG1G2(-dg, 0);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      auto mmm = ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom;

      cout << "MuG1 derivs: " << endl;
      dm = (mpp + mmm - mpm - mmp) / dg4dg;
      failure = compareMoments(derivOf(tmpl0, BC::DMU_DG1), dm) || failure;

      fac = 1+dg;
      tweak.setG1G2(0, dg);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      mpp = ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom;
      tweak.setG1G2(0, -dg);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      mpm = ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom;
      fac = 1-dg;
      tweak.setG1G2(0, dg);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      mmp = ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom;
      tweak.setG1G2(0, -dg);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      mmm = ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom;

      cout << "MuG2 derivs: " << endl;
      dm = (mpp + mmm - mpm - mmp) / dg4dg;
      failure = compareMoments(derivOf(tmpl0, BC::DMU_DG2), dm) || failure;
    }
      
    //*****************************************
    // Check moment and derivative rotation with rotated Gaussian
    //*****************************************
    double theta = 51. * PI / 180.;
    auto momR = mom0;
    momR.rotate(theta);

    // Create a rotated version of the galaxy:
    e = e0.getE();
    beta = e0.getBeta() + theta;
    Shear eR;
    eR.setEBeta(e,beta);
    Position<double> ctrR(x0 * cos(theta) - y0 * sin(theta),
			  x0 * sin(theta) + y0 * cos(theta));
    Ellipse ellR(eR, log(sigma), ctrR);
    auto ggR = ellipse2galaxy(wtSigma, flux, ellR, noise);

    cout << "Rotated moments vs rotated object: " << endl;
    failure = compareMoments(momR, ggR.getTarget().mom,"Moments","Object") || failure;
    
    // Make the two derivative sets
    auto tmpl1 = tmpl0;
    tmpl1.rotate(theta);
    auto tmpl2 = ggR.getTemplate();
    
    cout << "Rotated Template moments: " << endl;
    failure = compareMoments(derivOf(tmpl1,BC::P), derivOf(tmpl2,BC::P),
			     "Template","Object") || failure;
    cout << "Rotated dG1: " << endl;
    failure = compareMoments(derivOf(tmpl1,BC::DG1), derivOf(tmpl2,BC::DG1),
			     "Template","Object") || failure;
    cout << "Rotated dG2: " << endl;
    failure = compareMoments(derivOf(tmpl1,BC::DG2), derivOf(tmpl2,BC::DG2),
			     "Template","Object") || failure;
    if (BC::UseMag) {
      cout << "Rotated dMu: " << endl;
      failure = compareMoments(derivOf(tmpl1,BC::DMU), derivOf(tmpl2,BC::DMU),
			       "Template","Object") || failure;
    }
    cout << "Rotated dG1G1: " << endl;
    failure = compareMoments(derivOf(tmpl1,BC::DG1_DG1), derivOf(tmpl2,BC::DG1_DG1),
			     "Template","Object") || failure;
    cout << "Rotated dG1G2: " << endl;
    failure = compareMoments(derivOf(tmpl1,BC::DG1_DG2), derivOf(tmpl2,BC::DG1_DG2),
			     "Template","Object") || failure;
    cout << "Rotated dG2G2: " << endl;
    failure = compareMoments(derivOf(tmpl1,BC::DG2_DG2), derivOf(tmpl2,BC::DG2_DG2),
			     "Template","Object") || failure;
    if (BC::UseMag) {
      cout << "Rotated dMuG1: " << endl;
      failure = compareMoments(derivOf(tmpl1,BC::DMU_DG1), derivOf(tmpl2,BC::DMU_DG1),
			       "Template","Object") || failure;
      cout << "Rotated dMuG2: " << endl;
      failure = compareMoments(derivOf(tmpl1,BC::DMU_DG2), derivOf(tmpl2,BC::DMU_DG2),
			       "Template","Object") || failure;
      cout << "Rotated dMuMu: " << endl;
      failure = compareMoments(derivOf(tmpl1,BC::DMU_DMU), derivOf(tmpl2,BC::DMU_DMU),
			       "Template","Object") || failure;

    }

    //*****************************************
    // Check moment and derivative yflip with flipped Gaussian
    //*****************************************
    auto momF = mom0;
    momF.yflip();

    // Create a flipped version of the galaxy:
    e = e0.getE();
    beta = -e0.getBeta();
    Shear eF;
    eF.setEBeta(e,beta);
    Position<double> ctrF(x0,-y0);
    Ellipse ellF(eF, log(sigma), ctrF);
    auto ggF = ellipse2galaxy(wtSigma, flux, ellF, noise);

    cout << "Flipped moments vs flipped object: " << endl;
    failure = compareMoments(momF, ggF.getTarget().mom,
			     "Moments", "Object") || failure;
    
    // Make the two derivative sets
    tmpl1 = tmpl0;
    tmpl1.yflip();
    tmpl2 = ggF.getTemplate();
    
    cout << "Flipped Template moments: " << endl;
    failure = compareMoments(derivOf(tmpl1,BC::P), derivOf(tmpl2,BC::P),
			     "Template","Object") || failure;
    cout << "Flipped dG1: " << endl;
    failure = compareMoments(derivOf(tmpl1,BC::DG1), derivOf(tmpl2,BC::DG1),
			     "Template","Object") || failure;
    cout << "Flipped dG2: " << endl;
    failure = compareMoments(derivOf(tmpl1,BC::DG2), derivOf(tmpl2,BC::DG2),
			     "Template","Object") || failure;
    if (BC::UseMag) {
      cout << "Flipped dMu: " << endl;
      failure = compareMoments(derivOf(tmpl1,BC::DMU), derivOf(tmpl2,BC::DMU),
			       "Template","Object") || failure;
    }
    cout << "Flipped dG1G1: " << endl;
    failure = compareMoments(derivOf(tmpl1,BC::DG1_DG1), derivOf(tmpl2,BC::DG1_DG1),
			     "Template","Object") || failure;
    cout << "Flipped dG1G2: " << endl;
    failure = compareMoments(derivOf(tmpl1,BC::DG1_DG2), derivOf(tmpl2,BC::DG1_DG2),
			     "Template","Object") || failure;
    cout << "Flipped dG2G2: " << endl;
    failure = compareMoments(derivOf(tmpl1,BC::DG2_DG2), derivOf(tmpl2,BC::DG2_DG2),
			     "Template","Object") || failure;
    if (BC::UseMag) {
      cout << "Flipped dMuG1: " << endl;
      failure = compareMoments(derivOf(tmpl1,BC::DMU_DG1), derivOf(tmpl2,BC::DMU_DG1),
			       "Template","Object") || failure;
      cout << "Flipped dMuG2: " << endl;
      failure = compareMoments(derivOf(tmpl1,BC::DMU_DG2), derivOf(tmpl2,BC::DMU_DG2),
			       "Template","Object") || failure;
      cout << "Flipped dMuMu: " << endl;
      failure = compareMoments(derivOf(tmpl1,BC::DMU_DMU), derivOf(tmpl2,BC::DMU_DMU),
			       "Template","Object") || failure;

    }

    //*****************************************
    // Check that 2nd moments give Jacobian of 1st moments
    //*****************************************

    if (!BC::FixCenter) {
      // Create shifted versions of the galaxy
      double dx = 0.01;
      auto shiftp = gg0.getShifted(dx, 0.);
      auto shiftm = gg0.getShifted(-dx, 0.);
      typename BC::XYVector dxy = ( shiftp->getTarget().mom.xy - shiftm->getTarget().mom.xy) / (FP) (2.*dx);
      DVector theory(BC::XYSIZE);
      theory[BC::MX] = -0.5*(mom0.m[BC::MR]+mom0.m[BC::M1]);
      theory[BC::MY] = -0.5*mom0.m[BC::M2];
      cout << "d(MXY)/dx analytic vs object shift:" << endl;
      failure = compare(theory,DVector(dxy)) || failure;

      delete shiftp;
      delete shiftm;
      shiftp = gg0.getShifted(0., dx);
      shiftm = gg0.getShifted(0., -dx);
      dxy = ( shiftp->getTarget().mom.xy - shiftm->getTarget().mom.xy) / (FP) (2.*dx);
      theory[BC::MX] = -0.5*mom0.m[BC::M2];
      theory[BC::MY] = -0.5*(mom0.m[BC::MR]-mom0.m[BC::M1]);
      cout << "d(MXY)/dx analytic vs object shift:" << endl;
      failure = compare(theory,DVector(dxy)) || failure;

    }
    
    if (failure)
      cout << "****There were failures*****" << endl;
    exit(failure ? 1 : 0);

  } catch (std::runtime_error& m) {
    quit(m,1);
  }
}
