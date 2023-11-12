// Test the generation of multivariate Gaussian distribution
// in Distributions.h/cpp
#include <iostream>
#include "LinearAlgebra.h"
#include "Distributions.h"
#include "Random.h"

int
main(int argc,
     char *argv[])
{
  linalg::Matrix<double> cov(3,3);
  cov(0,0) = 4.;
  cov(1,1) = 6.;
  cov(2,2) = 3.;
  cov(1,0) = 2.;
  cov(2,0) = -0.5;
  cov(2,1) = -1.;

  int npts = 100000;
  linalg::Matrix<double> sum(3,3,0.);
  linalg::Vector<double> samp(3);

  bfd::MultiGauss<double> gg(cov);
  ran::GaussianDeviate<> gd;
  for (int i=0; i<npts; i++) {
    samp = gg.sample(gd);
    sum += samp.outer(samp);
  }
  sum /= (double) npts;
  cout << "Input: " << cov << endl;
  cout << "Output: " << sum << endl;
  exit(0);
}

  
