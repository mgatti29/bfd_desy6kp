// Test the derivatives of the PqrCalculations.
// ???Not testing the FixCenter versions of the selectors.

#include <algorithm>
#include "Shear.h"
#include "BfdConfig.h"
#include "GaussianGalaxy.h"
#include "PqrCalculation.h"
#include "Selection.h"
#include "testSubs.h"

using namespace bfd;
using namespace test;

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

GaussianGalaxy<BC> 
ellipse2galaxy(double wtSigma, double flux, const Ellipse& el, double noise) {
  double e =  el.getS().getE();
  double beta = el.getS().getBeta();
  Position<double> c = el.getX0();
  // The GaussianGalaxy wants sigma^2 to be the trace of the covariance matrix.
  double sigma = exp(el.getMu()) * pow(1-e*e, -0.25);
  return GaussianGalaxy<BC>(flux, sigma, e, beta, c.x, c.y, wtSigma, noise);
}

class StuffToTest {
public:
  StuffToTest(const TargetGalaxy<BC>& target,
	      const TemplateGalaxy<BC>& tmpl0):
    xyl(target), xyScale(1e5), ml(target), sel(0), asel(0), fsel(0), afsel(0) {
    // Initialize the classes under test
    // Save the Pqr for each one
    BC::MDMatrix dmG = tmpl0.realMDerivs();
    Pqr<BC> tmp(xyl(tmpl0.realXYDerivs())*xyScale);
    vpqr.push_back(tmp);
    
    vpqr.push_back(jac(dmG));

    // Move the target moments a bit for MLike testing
    BC::MVector tweakedTarget(target.mom.m);
    for (int i=0; i<BC::MSIZE; i++)
      tweakedTarget[i] += 0.1*sqrt(target.cov.m(i,i));
    ml.setTargetMoments(tweakedTarget);
    vpqr.push_back(ml(dmG));

    BC::FP dflux = sqrt(target.cov.m(BC::MF,BC::MF));
    BC::FP fluxmin = target.mom.m[BC::MF] - dflux;
    BC::FP fluxmax = target.mom.m[BC::MF] + dflux;
    sel = new Selector<BC>(fluxmin, fluxmax, target);
    vpqr.push_back(sel->probPqr(dmG));
    vpqr.push_back(sel->detectPqr(dmG));
    MomentCovariance<BC> addCov = 2 * target.cov;
    TargetGalaxy<BC> targAdded(target.mom, addCov);
    asel = new AddNoiseSelector<BC>(fluxmin, fluxmax, target, targAdded);
    vpqr.push_back(asel->probPqr(dmG));
    vpqr.push_back(asel->detectPqr(dmG));
    fsel = new FixedNoiseSelector<BC>(fluxmin, fluxmax, target);
    vpqr.push_back(fsel->probPqr(dmG));
    vpqr.push_back(fsel->detectPqr(dmG));
    afsel = new AddFixedNoiseSelector<BC>(fluxmin, fluxmax, target, targAdded);
    vpqr.push_back(afsel->probPqr(dmG));
    vpqr.push_back(afsel->detectPqr(dmG));
  }
  string banner() const {
    // Print out contents of columns
    return "        XYLike "
      " Jacobian "
      "   MLike  "
      " Sel/prob "
      " Sel/det  "
      " AddSel/p "
      " AddSel/d "
      " FixSel/p "
      " FixSel/d "
      " AFSel/p  "
      " AFSel/d  ";
  }
  DVector operator()(const Moment<BC>& mG) const {
    // Get the values at this galaxy point
    DVector out(11);
    out[0] = xyl(mG.xy) * xyScale;
    out[1] = jac(mG.m);
    out[2] = ml(mG.m);
    out[3] = sel->prob(mG);
    out[4] = sel->detect(mG);
    out[5] = asel->prob(mG);
    out[6] = asel->detect(mG);
    out[7] = fsel->prob(mG);
    out[8] = fsel->detect(mG);
    out[9] = afsel->prob(mG);
    out[10] = afsel->detect(mG);
    return out;
  }
  DVector derivs(int i) {
    // get requested derivative for all test subjects
    DVector out(vpqr.size());
    for (int j=0; j<vpqr.size(); j++)
      out[j] = vpqr[j][i];
    return out;
  }
  ~StuffToTest() {
    if (sel) delete sel;
    if (asel) delete asel;
    if (fsel) delete fsel;
    if (afsel) delete afsel;
  }
private:
  XYLike<BC> xyl;
  XYJacobian<BC> jac;
  MLike<BC> ml;
  vector<Pqr<BC>> vpqr;
  BC::FP xyScale;
  Selector<BC> *sel;
  AddNoiseSelector<BC> *asel;
  FixedNoiseSelector<BC> *fsel;
  AddFixedNoiseSelector<BC> *afsel;
};

int main(int argc,
	 char *argv[])
{
  try {
    double sn = 10.;
    double noise = 1.;
    // Intrinsic galaxy shape
    double sigma = 1.5;
    double wtSigma = 1.;
    double e=0.;
    double beta=0.;
    double x0=0.;
    double y0=0.;

    if (argc<2 || argc > 8) {
      x0 = -0.3;
      y0 = 0.4;
      sn = 10.;
      sigma = 2.0;
      e = 0.6;
      wtSigma = 0.5;
      beta = 140. * 3.1415/180.;

      cerr 
	<< "Compare analytic derivatives of XYLike, xyJacobian\n"
	" w.r.t. g1,g2,mu to finite differences from same classes.\n"
	"Usage: testPqrCalculations <sn> [sigma=1.5] [e=0.] [wtSigma=1.] [beta=0.] [x0=0.] [y0=0.]\n"
	" where you give values but will take noted default values starting at rhs if not all args are\n"
	" given.\n"
	" stdout will be comparison of results.\n"
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

    bool failure = false;

    // Get the baseline analytic galaxy
    double e1 = e * cos(2*beta);
    double e2 = e * sin(2*beta);
    Shear e0(e1, e2);
    Position<double> ctr(x0,y0); 
    double flux = sn * sqrt(4*PI*sigma*sigma*noise);
    Ellipse ell0(e0, log(sigma), ctr);
    GaussianGalaxy<BC> gg0 = ellipse2galaxy(wtSigma, flux, ell0, noise);
    auto targ0 = gg0.getTarget(true);
    auto mom0 = targ0.mom;  // Moment structure

    StuffToTest stt(targ0, gg0.getTemplate());
    // Unlensed values:
    auto v0 = stt(targ0.mom);
    
    //*****************************************
    // Check analytic derivative vs finite difference
    //*****************************************

    // Now check finite differences of shear and magnification
    const FP dg = 0.002;
    const FP TWO = 2;

    Shear tweak;
    tweak.setG1G2(dg, 0.);
    Ellipse dEl(tweak, 0., Position<double>(0.,0.));
    auto vg1p = stt(ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget(false).mom);
    tweak.setG1G2(-dg, 0.);
    dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
    auto vg1m = stt(ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget(false).mom);
    tweak.setG1G2(0., dg);
    dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
    auto vg2p = stt(ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget(false).mom);
    tweak.setG1G2(0.,-dg);
    dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
    auto vg2m = stt(ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget(false).mom);

    DVector vmup(v0), vmum(v0);
    if (BC::UseMag) {
      double fac = 1+dg;
      dEl = Ellipse(Shear(), log(fac), Position<double>(0.,0.));
      vmup = stt(ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom);
      fac = 1-dg;
      dEl = Ellipse(Shear(), log(fac), Position<double>(0.,0.));
      vmum = stt(ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom);
    }

    // Print out the column headings
    cout << stt.banner() <<endl;
    
    // First compare the Pqr value to straight-up
    cout << "Value:" << endl;
    failure = compare(stt.derivs(BC::P), v0, "Pqr", "Direct") || failure;
      
    // Then derivatives
    
    DVector dv(v0);
    cout << "G1 derivs: " << endl;
    dv = (vg1p-vg1m) / (TWO*dg);
    failure = compare(stt.derivs(BC::DG1),dv) || failure;

    cout << "G2 derivs: " << endl;
    dv = (vg2p-vg2m) / (TWO*dg);
    failure = compare(stt.derivs(BC::DG2),dv) || failure;
    
    if (BC::UseMag) {
      cout << "Mu derivs: " << endl;
      dv = (vmup-vmum) / (TWO*dg);
      failure = compare(stt.derivs(BC::DMU),dv) || failure;
    }
    
    cout << "G1G1 derivs: " << endl;
    dv = (vg1p + vg1m - TWO*v0) / (dg*dg);
    failure = compare(stt.derivs(BC::DG1_DG1),dv) || failure;

    cout << "G2G2 derivs: " << endl;
    dv = (vg2p + vg2m - TWO*v0) / (dg*dg);
    failure = compare(stt.derivs(BC::DG2_DG2),dv) || failure;

    if (BC::UseMag) {
      cout << "MuMu derivs: " << endl;
      dv = (vmup + vmum - TWO*v0) / (dg*dg);
      failure = compare(stt.derivs(BC::DMU_DMU),dv) || failure;
    }

    // Crossed 2nd derivatives:
    const FP dg4dg=4.*dg*dg;
    {
      tweak.setG1G2(dg, dg);
      dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
      auto vpp = stt(ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget().mom);
      tweak.setG1G2(dg, -dg);
      dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
      auto vpm = stt(ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget().mom);
      tweak.setG1G2(-dg, dg);
      dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
      auto vmp = stt(ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget().mom);
      tweak.setG1G2(-dg, -dg);
      dEl = Ellipse(tweak, 0., Position<double>(0.,0.));
      auto vmm = stt(ellipse2galaxy(wtSigma, flux, dEl + ell0, noise).getTarget().mom);

      cout << "G1G2 derivs: " << endl;
      dv = (vpp + vmm - vpm - vmp) / dg4dg;
      failure = compare(stt.derivs(BC::DG1_DG2),dv) || failure;
    }

    if (BC::UseMag) {
      double fac = 1+dg;
      tweak.setG1G2(dg, 0);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      auto vpp = stt(ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom);
      tweak.setG1G2(-dg, 0);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      auto vpm = stt(ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom);
      fac = 1-dg;
      tweak.setG1G2(dg, 0);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      auto vmp = stt(ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom);
      tweak.setG1G2(-dg, 0);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      auto vmm = stt(ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom);

      cout << "MuG1 derivs: " << endl;
      dv = (vpp + vmm - vpm - vmp) / dg4dg;
      failure = compare(stt.derivs(BC::DMU_DG1),dv) || failure;

      fac = 1+dg;
      tweak.setG1G2(0, dg);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      vpp = stt(ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom);
      tweak.setG1G2(0, -dg);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      vpm = stt(ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom);
      fac = 1-dg;
      tweak.setG1G2(0, dg);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      vmp = stt(ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom);
      tweak.setG1G2(0, -dg);
      dEl = Ellipse(tweak, log(fac), Position<double>(0.,0.));
      vmm = stt(ellipse2galaxy(wtSigma, flux*fac*fac, dEl+ell0, noise).getTarget(false).mom);

      cout << "MuG2 derivs: " << endl;
      dv = (vpp + vmm - vpm - vmp) / dg4dg;
      failure = compare(stt.derivs(BC::DMU_DG2),dv) || failure;
    }
    
    if (failure)
      cout << "****There were failures*****" << endl;
    exit(failure ? 1 : 0);

  } catch (std::runtime_error& m) {
    quit(m,1);
  }
}
