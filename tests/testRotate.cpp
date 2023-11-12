// See if I have the Pqr rotation correct
#include "BfdConfig.h"
#include "Pqr.h"
#include "testSubs.h"

#include <iostream>

using namespace bfd;
using namespace std;

// Do the test with all derivatives
const bool FIX_CENTER=false;
const bool USE_MAG = true;
const bool USE_CONC = true;
const int N_COLORS=0;
const bool USE_FLOAT=true;
typedef BfdConfig<FIX_CENTER,
		  USE_CONC,
		  USE_MAG,
		  N_COLORS,
		  USE_FLOAT> BC;

int main(int argc,
	 char *argv[])
{
  // Define a lensing distortion
  double g1 = 0.2;
  double g2 = -0.4;
  double mu = 0.11;

  BC::QVector lens0;
  lens0[BC::G1] = g1;
  lens0[BC::G2] = g2;
  lens0[BC::MU] = mu;

  // Define some Taylor expansion wrt lensing
  Pqr<BC> p;
  p[BC::P] = 0.33;
  p[BC::DG1] = -0.73;
  p[BC::DG2] = 0.1;
  p[BC::DMU] = 0.1;
  p[BC::DG1_DG1] = 1.3;
  p[BC::DG1_DG2] = -2.;
  p[BC::DG2_DG2] = 1.7;
  p[BC::DMU_DMU] = 2.1;
  p[BC::DMU_DG1] = -2.3;
  p[BC::DMU_DG2] = -0.4;

  double original = p(lens0);
  
  // Now define same lensing in coordinate system rotated by -theta
  double beta = 0.5;
  double c = cos(2*beta);
  double s = sin(2*beta);
  BC::QVector lens1;
  lens1[BC::G1] = g1*c - g2*s;
  lens1[BC::G2] = g1*s + g2*c;
  lens1[BC::MU] = mu;

  p.rotate(beta);
  double rotated = p(lens1);

  bool failure = test::compare(original, rotated, "Original", "Rotated");

  if (failure) {
    cout << "FAILURE" << endl;
    exit(1);
  } else {
    exit(0);
  }
}

