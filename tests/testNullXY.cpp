// Test the iterative nulling of XY moments using a GaussianGalaxy
// and also test some GalaxyPlusNoise behavior.

#include <algorithm>
#include "Random.h"
#include "BfdConfig.h"
#include "Galaxy.h"
#include "GaussianGalaxy.h"
#include "testSubs.h"
#include "Shear.h"

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

GaussianGalaxy<BC> 
ellipse2galaxy(double wtSigma, double flux, const Ellipse& el, double noise) {
  double e =  el.getS().getE();
  double beta = el.getS().getBeta();
  Position<double> c = el.getX0();
  // The GaussianGalaxy wants sigma^2 to be the trace of the covariance matrix.
  double sigma = exp(el.getMu()) * pow(1-e*e, -0.25);
  return GaussianGalaxy<BC>(flux, sigma, e, beta, c.x, c.y, wtSigma, noise);
}


int main(int argc,
	 char *argv[])
{
  try {
    double sn = 20.;
    double x0 = -0.25;
    double y0 = 0.3;
    double sigma = 2.0;
    double e = 0.6;
    double wtSigma = 2.0;
    double beta = 140. * 3.1415/180.;
    double noise = 1.;

    if (argc<2 || argc > 2) {

      cerr 
	<< "Test iterative shifting to null XY moments.\n"
	"Usage: testNullXY [sn=20] \n" 
	"First test tries to recenter an off-center noiseless Gaussian.\n"
	"Second test adds noise to these moments are recenters.  Do not\n"
	"expect convergence at low S/N."
	<< endl;
    } 
    if (argc > 1) sn  = atof(argv[1]);

    bool failure = false;
    
    // Get the analytic galaxy moments
    double e1 = e * cos(2*beta);
    double e2 = e * sin(2*beta);
    Shear e0(e1, e2);
    Position<double> ctr(x0,y0); 
    double flux = sn * sqrt(4*PI*sigma*sigma*noise);
    Ellipse ell0(e0, log(sigma), ctr);
    GaussianGalaxy<BC> gg0 = ellipse2galaxy(wtSigma, flux, ell0, noise);

    cout << "Noiseless getNullXY" << endl;
    GalaxyData<BC>* centered = gg0.getNullXY(2.);
    auto gg1 = dynamic_cast<GaussianGalaxy<BC>*> (centered);
    if (!centered) {
      cout << "-------------> FAILURE" << endl;
      cout << "Nulling did not converge" << endl;
      failure = true;
      // Do not proceed if we don't have a centered galaxy.
      cout << "****There were failures*****" << endl;
      exit(1);
    } else if (!gg1) {
      cout << "-------------> FAILURE" << endl;
      cout << "getNullXY did not return GaussianGalaxy" << endl;
      failure = true;
    } else {
      gg1->dump(cout);
      cout << endl;
      failure = test::compare(centered->getTarget(false).mom.xy,
			      DVector2(0.), "Recentered", "Null") || failure;
    }
    
    // Make a noisy version of the centered galaxy and recenter it.
    ran::UniformDeviate<double> ud(363L); // Choose a seed.
    ran::GaussianDeviate<double> gd(ud);
    GalaxyPlusNoise<BC> gnoise(centered,gd);
    /**
    cout << "Moment sigma: " << sqrt(gnoise.getTarget(true).cov.xy(BC::MX,BC::MX))
	 << endl;
    **/
    cout << "\nNoisy getNullXY" << endl;
    auto centered2 = gnoise.getNullXY(6.);
    auto gg2 = dynamic_cast<GalaxyPlusNoise<BC>*> (centered2);
    if (!centered2) {
      cout << "-------------> FAILURE" << endl;
      cout << "Nulling did not converge" << endl;
      failure = true;
    } else if (!gg2) {
      cout << "-------------> FAILURE" << endl;
      cout << "getNullXY did not return GalaxyPlusNoise" << endl;
      failure = true;
    } else {
      failure = test::compare(centered2->getTarget(false).mom.xy,
			      DVector2(0.), "Recentered", "Null") || failure;
    }
    if (centered) delete centered;

    if (failure)
      cout << "****There were failures*****" << endl;
    exit(failure ? 1 : 0);

  } catch (std::runtime_error& m) {
    quit(m,1);
  }
}
