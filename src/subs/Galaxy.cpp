// Transformations for moment derivatives
#include <complex>
#include "Galaxy.h"
#include <set>
#include <utility>

using namespace bfd;

void
test(const MomentCovariance<BfdConfig<>>& m) {
  MomentCovariance<BfdConfig<>> m2;
  m2 = m;
}

template <class CONFIG>
void
TemplateGalaxy<CONFIG>::rotate(double theta) {
  const int maxM = 4;
  linalg::SVector<FPC,maxM+1 > eintheta;  // exp(i n theta) array
  FPC eitheta(cos(theta), sin(theta));
  eintheta[0] = 1.;
  for (int i=1; i<=maxM; i++) eintheta[i] = eintheta[i-1]*eitheta;

  // First do the even moments:
  // Factor for the only non-monopole moment
  mDeriv.row(ME) *= eintheta[2];   
  // Factors for the derivatives
  mDeriv.col(DV) *= conj(eintheta[2]);  
  mDeriv.col(DVb) *= eintheta[2];
  if (BC::UseMag) {
    mDeriv.col(DU_DV) *= conj(eintheta[2]);
    mDeriv.col(DU_DVb) *= eintheta[2];
  }
  mDeriv.col(DV_DV) *= conj(eintheta[4]);
  mDeriv.col(DVb_DVb) *= eintheta[4];

  if (!BC::FixCenter) {
    // Do the odd moments too
    xyDeriv.row(MX) *= eintheta[1];   
    // Factors for the derivatives
    xyDeriv.col(DV) *= conj(eintheta[2]);  
    xyDeriv.col(DVb) *= eintheta[2];
    if (BC::UseMag) {
      xyDeriv.col(DU_DV) *= conj(eintheta[2]);
      xyDeriv.col(DU_DVb) *= eintheta[2];
    }
    xyDeriv.col(DV_DV) *= conj(eintheta[4]);
    xyDeriv.col(DVb_DVb) *= eintheta[4];
  }
  return;
}

template <class CONFIG>
void
TemplateGalaxy<CONFIG>::yflip() {
  // All even moments have their imaginary parts inverted
  mDeriv.IMAG *= FP(-1.);
  // And odd moments have their imag parts inverted too
  if (!BC::FixCenter) {
    xyDeriv.IMAG *= FP(-1.);
  }
  return;
}

template <class CONFIG>
typename CONFIG::MDMatrix 
TemplateGalaxy<CONFIG>::realMDerivs() const {
  typename BC::MDMatrix out( FP(0.));
  FPC I(FP(0.),FP(1.));
  // Convert the DU / DV / DVb derivatives into DMU / DG1 / DG2 first
  DerivMatrix mtmp(MSIZE,DSIZE,0.);
  mtmp.col(BC::P) = mDeriv.col(D0);
  mtmp.col(BC::DG1) = mDeriv.col(DV) + mDeriv.col(DVb);
  mtmp.col(BC::DG2) = I * (mDeriv.col(DV) - mDeriv.col(DVb));
  mtmp.col(BC::DG1_DG1) = FP(2.) * mDeriv.col(DV_DVb) + mDeriv.col(DV_DV) +
    mDeriv.col(DVb_DVb);
  mtmp.col(BC::DG2_DG2) = FP(2.) * mDeriv.col(DV_DVb) - mDeriv.col(DV_DV) -
    mDeriv.col(DVb_DVb);
  mtmp.col(BC::DG1_DG2) = I * (mDeriv.col(DV_DV) - mDeriv.col(DVb_DVb));
  if (BC::UseMag) {
    mtmp.col(BC::DMU) = mDeriv.col(DU);
    mtmp.col(BC::DMU_DG1) = mDeriv.col(DU_DV) + mDeriv.col(DU_DVb);
    mtmp.col(BC::DMU_DG2) = I * (mDeriv.col(DU_DV) - mDeriv.col(DU_DVb));
    mtmp.col(BC::DMU_DMU) = mDeriv.col(DU_DU);
  }

  // Then extract appropriate parts into real moments
  out.row(BC::MF) = mtmp.row(MF).REAL;
  out.row(BC::MR) = mtmp.row(MR).REAL;
  out.row(BC::M1) = mtmp.row(ME).REAL;
  out.row(BC::M2) = mtmp.row(ME).IMAG;
  if (BC::UseConc) out.row(BC::MC) = mtmp.row(MC).REAL;
  for (int i=0; i<BC::Colors; i++)
    out.row(BC::MC0+i) = mtmp.row(MC0+i).REAL;
  return out;
}
template <class CONFIG>
typename CONFIG::XYDMatrix 
TemplateGalaxy<CONFIG>::realXYDerivs() const {
  typename BC::XYDMatrix out( FP(0.));
  const FPC I(FP(0.),FP(1.));
  if (BC::FixCenter) return out;  // Return zeros if this is irrelevant

  // Convert the DU / DV / DVb derivatives into DMU / DG1 / DG2 first
  DerivMatrix xytmp(XYSIZE,DSIZE,0.);
  xytmp.col(BC::P) = xyDeriv.col(D0);
  xytmp.col(BC::DG1) = xyDeriv.col(DV) + xyDeriv.col(DVb);
  xytmp.col(BC::DG2) = I * (xyDeriv.col(DV) - xyDeriv.col(DVb));
  xytmp.col(BC::DG1_DG1) = FP(2.) * xyDeriv.col(DV_DVb) + xyDeriv.col(DV_DV) +
    xyDeriv.col(DVb_DVb);
  xytmp.col(BC::DG2_DG2) = FP(2.) * xyDeriv.col(DV_DVb) - xyDeriv.col(DV_DV) -
    xyDeriv.col(DVb_DVb);
  xytmp.col(BC::DG1_DG2) = I * (xyDeriv.col(DV_DV) - xyDeriv.col(DVb_DVb));
  if (BC::UseMag) {
    xytmp.col(BC::DMU) = xyDeriv.col(DU);
    xytmp.col(BC::DMU_DG1) = xyDeriv.col(DU_DV) + xyDeriv.col(DU_DVb);
    xytmp.col(BC::DMU_DG2) = I * (xyDeriv.col(DU_DV) - xyDeriv.col(DU_DVb));
    xytmp.col(BC::DMU_DMU) = xyDeriv.col(DU_DU);
  }

  // Then extract appropriate parts into real moments
  out.row(BC::MX) = xytmp.row(MX).REAL;
  out.row(BC::MY) = xytmp.row(MX).IMAG;
  return out;
}

///////////////////////////////////////////////////////////////
// Construct the added-noise class
///////////////////////////////////////////////////////////////
template <class CONFIG>
GalaxyPlusNoise<CONFIG>::GalaxyPlusNoise(const GalaxyData<BC>* gptr_,
					 ran::GaussianDeviate<double>& gd,
					 const MultiGauss<FP>* noiseGen):
  gptr(gptr_)
{
  typename BC::Vector mall(BC::MXYSIZE,0.); // temp place for the noise vector
  if (noiseGen) {
#ifdef _OPENMP
#pragma omp critical(random)
#endif
    mall = noiseGen->sample(gd);  // Only one thread drawing random numbers at a time
    
    auto covall = noiseGen->cov();
    Assert(mall.size()==BC::MXYSIZE); // Generator should make total moments
    noise.m = mall.subVector(0,BC::MSIZE);
    cov.m = covall.subMatrix(0,BC::MSIZE,0,BC::MSIZE);
    /**
    for (int i=0; i<BC::MSIZE; i++) {
      noise.m[i] = mall[i];
      for (int j=0; j<BC::MSIZE; j++)
	cov.m(i,j) = covall(i,j);
	}**/
    if (!BC::FixCenter) {
      noise.xy = mall.subVector(BC::MSIZE,BC::MXYSIZE);
      cov.xy = covall.subMatrix(BC::MSIZE,BC::MXYSIZE,BC::MSIZE,BC::MXYSIZE);
      /**
      for (int i=0; i<BC::XYSIZE; i++) {
	noise.xy[i] = mall[i+BC::MSIZE];
	for (int j=0; j<BC::XYSIZE; j++)
	  cov.xy(i,j) = covall(i+BC::MSIZE,j+BC::MSIZE);
	  }**/
    }
  } else {
    // Make a noise generator from the covariance matrix
    // and draw once from it
    cov = gptr->getTarget().cov;
    typename BC::Matrix covall(BC::MXYSIZE,BC::MXYSIZE,FP(0.));
    covall.subMatrix(0,BC::MSIZE,0,BC::MSIZE) = cov.m;
    if (!BC::FixCenter)
      covall.subMatrix(BC::MSIZE,BC::MXYSIZE,BC::MSIZE,BC::MXYSIZE) = cov.xy;
    
    /**
    for (int i=0; i<BC::MSIZE; i++)
      for (int j=0; j<=BC::MSIZE; j++)
	covall(i,j) = cov.m(i,j);
    if (!BC::FixCenter)
      for (int i=0; i<BC::XYSIZE; i++)
	for (int j=0; j<=i; j++)
	  symcov(i+BC::MSIZE,j+BC::MSIZE) = cov.xy(i,j);
    **/
    MultiGauss<FP> gen(covall);
#ifdef _OPENMP
#pragma omp critical(random)
#endif
    mall = gen.sample(gd);
    noise.m = mall.subVector(0,BC::MSIZE);
    if (!BC::FixCenter)
      noise.xy = mall.subVector(BC::MSIZE,BC::MXYSIZE);
    /**
    for (int i=0; i<BC::MSIZE; i++)
      noise.m[i] = mall[i];
    if (!BC::FixCenter)
      for (int i=0; i<BC::XYSIZE; i++)
	noise.xy[i] = mall[i+BC::MSIZE];
	}**/
  }
}

///////////////////////////////////////////////////////////////
// Generic GalaxyData manipulations 
///////////////////////////////////////////////////////////////

template <class CONFIG>
typename CONFIG::XYMatrix
GalaxyData<CONFIG>::xyJacobian() const {
  typename BC::XYMatrix out;
  if (BC::FixCenter) return out;
  auto m = getTarget(false).mom.m;
  out(BC::MX,BC::MX) = FP(0.5) * (m[BC::MR]+m[BC::M1]);
  out(BC::MY,BC::MY) = FP(0.5) * (m[BC::MR]-m[BC::M1]);
  out(BC::MX,BC::MY) = FP(0.5) * m[BC::M2];
  out(BC::MY,BC::MX) = FP(0.5) * m[BC::M2];
  return out;
}
  

template <class CONFIG>
GalaxyData<CONFIG>*
GalaxyData<CONFIG>::getNullXY(FP maxShift) const {
  // We'll do a Newton iteration a minimum of this many times:
  const int MIN_ITERATIONS = 3;
  // And stop when the XY centroid moments are both smaller than
  // this quantity times the smaller SV of the Jacobian derivative
  // times the maxShift:
  const double TOL = 1e-5;
  // Returns null if we get outside of maxShift or reach this many iterations
  const int MAX_ITERATIONS = 10;
  
  double dx=0., dy = 0.;
  if (BC::FixCenter) {
    // Not using centroid, just return a copy of input
    return getShifted(dx,dy);
  }

  auto mxy = getTarget(false).mom.xy;
  GalaxyData<BC>* out = 0;  // The current shifted template
  for (int iter=0; iter<MAX_ITERATIONS; iter++) {
    // Test for convergence
    typename BC::XYMatrix jac = (out) ? out->xyJacobian() : xyJacobian();
    // Get the smaller singular value of the Jacobian
    double maxdmdx;
    {
      double a = jac(BC::MX,BC::MX);
      double b = jac(BC::MX,BC::MY);
      double c = jac(BC::MY,BC::MX);
      double d = jac(BC::MY,BC::MY);
      double S1 = a*a + b*b + c*c + d*d;
      double S2 = a*a + b*b - c*c - d*d;
      S2 = sqrt( (S2*S2 + 4*(a*c+b*d)*(a*c+b*d)));
      maxdmdx = sqrt( (S1 - S2)/2.);
    }
    double maxM = maxShift * maxdmdx;
    if (iter >= MIN_ITERATIONS &&
	hypot(mxy[BC::MX],mxy[BC::MY]) < TOL * maxM) {
      // Converged!
      return out;
    }
    // Calculate & apply Newton step
    typename BC::XYVector dxy(jac.inverse() * mxy);
    dx += dxy[BC::MX];
    dy += dxy[BC::MY];
    /**cerr << iter << " dxy " << dxy
	     << " mxy " << mxy << " dx dy " << dx << " " << dy << endl;
    **/
    if (hypot(dx,dy) > maxShift) {
      // Wandered too far
      if (out) delete out;
      return 0;
    }
    // Get shifted galaxy
    out = getShifted(dx, dy);
    mxy = out->getTarget(false).mom.xy;    
  }
  // If we get here we've exceed MAX_ITERATIONS
  if (out) delete out;
  return 0;
}

// These classes will be used in creation of a grid of
// templates in XY space:
typedef std::pair<int,int> GridPoint;
class GridList: public std::set<GridPoint> {
public:
  GridPoint pop() {
    GridPoint out = *begin();
    erase(begin());
    return out;
  }
  bool has(GridPoint ij) const {
    return find(ij)!=end();
  }
};


template <class CONFIG>
vector<TemplateGalaxy<CONFIG>>
GalaxyData<CONFIG>::getTemplateGrid(ran::UniformDeviate<double>& ud,
				    FP xySigma,
				    FP fluxSigma,
				    FP fluxMin,
				    FP sigmaStep,
				    FP sigmaMax,
				    FP xyMax) const {
  vector<TemplateGalaxy<BC>> out;
  if (BC::FixCenter) {
    // Return a single template if there is no shifting
    out.push_back(getTemplate());
    return out;
  }

  // Get the centered version of this galaxy:
  auto centered = getNullXY(xyMax);
  // And its Jacobian and determinant
  if (!centered) {
    // Centering failed.  Return empty grid
    return out;
  }
  auto j0Mat = centered->xyJacobian();
  FP j0Det = j0Mat.det();
  // The eigenvectors of the Jacobian will be the
  // directions in which we build our XY grid
  linalg::DMatrix22 evecs;
  linalg::DVector2 evals;

  {
    // Get the eigenvectors / values of 2x2 symmetric matrix
    FP x = sqrt( SQR(j0Mat(0,0)-j0Mat(1,1)) + 4 * SQR(j0Mat(0,1)));
    evals[0] = (j0Mat(0,0) + j0Mat(1,1) + x) / 2;
    evals[1] = (j0Mat(0,0) + j0Mat(1,1) - x) / 2;
    FP csq = 1.;
    if (x>0.)
      csq = 0.5 * (1. + (j0Mat(0,0)-j0Mat(1,1))/x);
    evecs(0,0) = evecs(1,1) = sqrt(csq);
    evecs(1,0) = evecs(0,1) = sqrt(1.-csq);
    if (j0Mat(1,0)>0) evecs(0,1) *=-1.;
    else evecs(1,0) *=-1.;
  }

  if (evals[0]<0 || evals[1] < 0) {
    cerr << "WARNING: Galaxy not at flux maximum in getTemplateGrid" << endl;
    cerr << "  Eigenvalues " << evals << endl;
    return out;
  }
  // Set grid step sizes to change moments by sigmaStep times noise
  linalg::DVector2 xyStep;
  for (int i=0; i<2; i++)
    xyStep[i] = xySigma * sigmaStep / evals[i];
  // Grid cell area factor:
  FP dA = xyStep[0] * xyStep[1];
  
  // Offset the grid by a random amount
  linalg::DVector2 xyOffset;
#ifdef _OPENMP
#pragma omp critical(random)
#endif
  {
    for (int i=0; i<2; i++)
      xyOffset[i] = ud() - 0.5;
  }

  // Make a container of grid points to try and another of points
  // already done.
  GridList toTry;
  GridList done;
  // Initiate the grid search at the origin
  toTry.insert(GridPoint(0,0));

  // Try grid points until all viable ones are gone
  while (!toTry.empty()) {
    GridPoint ij = toTry.pop();
    int i = ij.first;
    int j = ij.second;
    done.insert(ij);
    
    linalg::DVector2 dij(xyOffset);
    dij[0] += i;
    dij[1] += j;
    dij[0] *= xyStep[0];
    dij[1] *= xyStep[1];
    
    linalg::DVector2 xy(evecs * dij);  // ??? ordering ???
    if ( xy.dot(xy) > xyMax * xyMax) {
      // This point is out of bounds, so skip it.
      continue;
    }

    // Make galaxy shifted to this point
    auto shifted = centered->getShifted(xy[0],xy[1]);
    // Get its moments
    Moment<BC> mom = shifted->getTarget(false).mom;

    // We will not use regions where the map from x,y to MX,MY moments is no longer
    // 1-to-1 as indicated by a non-positive Jacobian:
    FP jDet = 0.25 * (mom.m[BC::MR]*mom.m[BC::MR]
		      - mom.m[BC::M1]*mom.m[BC::M1]
		      - mom.m[BC::M2]*mom.m[BC::M2]);

    if (jDet < 0.) {
      // ?? Any kind of warning or flag for Jacobian sign flip ??
      delete shifted;
      continue;
    }

    // Calculate its minimal distance of this template from a target:
    // First the X/Y moment contribution
    FP chisq = mom.xy.dot(mom.xy) / (xySigma*xySigma);
    // Then suppression by small jacobian; add the factor to "chisq":
    chisq += -2. * log(jDet / j0Det);
    // Then a factor if its flux is below cutoff
    if (mom.m[BC::MF] < fluxMin) {
      chisq += SQR( (fluxMin-mom.m[BC::MF])/fluxSigma);
    } 

    if (chisq <= sigmaMax*sigmaMax) {
      // Valid point.  Keep its template
      out.push_back(shifted->getTemplate());
      // Update its area factor and Jacobian suppresion
      out.back().nda *= dA;
      out.back().jSuppression = jDet / j0Det;
      // Add neighboring grid points to list of ones to try if they
      // are not already done.
      for (int ii : {ij.first-1, ij.first+1}) {
	// To left & right:
	GridPoint trial(ii,ij.second);
	if (!done.has(trial)) toTry.insert(trial);
      }
      for (int jj : {ij.second-1, ij.second+1}) {
	// Above/below:
	GridPoint trial(ij.first,jj);
	if (!done.has(trial)) toTry.insert(trial);
      }
    }
    delete shifted;
  } // end grid point loop
  return out;
}

///////////////////////////////////////////////////////////////
// 
// Instantiations of the templates
//
///////////////////////////////////////////////////////////////

#define INSTANTIATE(...)	\
  template class bfd::TemplateGalaxy<BfdConfig<__VA_ARGS__> >;	\
  template class bfd::GalaxyData<BfdConfig<__VA_ARGS__> >;     \
  template class bfd::GalaxyPlusNoise<BfdConfig<__VA_ARGS__> >;

#include "InstantiateMomentCases.h"

