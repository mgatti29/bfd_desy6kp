#include "Prior.h"

using namespace bfd;

template<class CONFIG>
Prior<CONFIG>:: Prior(FP fluxMin_,
		      FP fluxMax_,
		      const MomentCovariance<BC>& nominalCov,
		      ran::UniformDeviate<double>& ud_,
		      bool selectionOnly_,
		      FP noiseFactor_,
		      FP sigmaStep_,
		      FP sigmaCutoff_,
		      bool invariantCovariance_,
		      bool fixedNoise_):
  fluxMin(fluxMin_),
  fluxMax(fluxMax_),
  ndaTotal(FP(0)),
  nominalSelector(0),
  noiseFactor(noiseFactor_),
  addedCovariance(FP(0)),
  addedNoiseGenerator(0),
  ud(ud_),
  gd(0),
  invariantCovariance(invariantCovariance_),
  selectionTemplateCounter(0),
  fixedNoise(fixedNoise_ && !BC::FixCenter),  // Fixed noise irrelevant with fixed center
  isPrepared(false),
  selectionOnly(selectionOnly_),
  sigmaCutoff(sigmaCutoff_),
  sigmaStep(sigmaStep_)
{
  if (selectionOnly) {
    nominalCovariance = nominalCov;
  } else {
    // isotropize the nominal covariance if not doing selection
    nominalCovariance = nominalCov.isotropize();
    nominalTotalCovariance = nominalCovariance;
    
    // Make sure cov matrix is rotationally invariant if invariantCovariance=true
    if (invariantCovariance) {
      if (!nominalCov.isIsotropic()) {
	invariantCovariance = false;
	cerr << "**WARNING: Shutting off invariantCovariance in Prior because\n"
	        "**WARNING: input nominal covariance is anisotropic." << endl;
      }
    }
    // Set up for adding noise if desired
    if (noiseFactor>1.) setupAddedNoise();
  }
  // Build the nominalSelector, which just needs the nominal covariance w/o added noise
  TargetGalaxy<BC> dummyTarget( Moment<BC>(),
				nominalCovariance);
  nominalSelector = chooseSelector(dummyTarget, dummyTarget);
}


template<class CONFIG>
void 
Prior<CONFIG>::setUniqueCeiling(int ceiling) {
  PqrAccumulator<BC>::idCeiling = ceiling;
}

template <class CONFIG>
void
Prior<CONFIG>::setupAddedNoise() {

  // The nominalCovariance is assumed to be ALREADY ISOTROPIC and
  // to satisfy the MR-M1-M2=0 condition.

  // No need of this if no added noise or if we are only calculating selection
  // factors.
  if (noiseFactor<=1. || selectionOnly) return;

  // We'll need a Gaussian deviate to generate noise
  if (gd) delete gd;
  gd = new ran::GaussianDeviate<double>(ud);

  // Make an isotropized covariance matrix for the added noise.  
  addedCovariance = nominalCovariance.m * FP(noiseFactor * noiseFactor - 1.);

  // create a noise generator for it, kill any existing ones
  if (addedNoiseGenerator) delete addedNoiseGenerator;

  // Need to convert from MomentCovariance to SymMatrix, and skip the degenerate rows
  // Use all dimensions:
  /**
  typename BC::SymMatrix covtmp(BC::MSIZE);
  for (int i=0; i<BC::MSIZE; i++)
    for (int j=0; j<BC::MSIZE; j++)
      covtmp(i,j) = addedCovariance(i,j);
  addedNoiseGenerator = new MultiGauss<FP>(covtmp);
  **/
  addedNoiseGenerator = new MultiGauss<FP>(addedCovariance);

  nominalTotalCovariance.m = nominalCovariance.m + addedCovariance;
  nominalTotalCovariance.xy = nominalCovariance.xy;
  return;
}

template<class CONFIG>
void
Prior<CONFIG>::addNoiseTo(TargetGalaxy<BC>& targ) const {
  TargetGalaxy<BC> targAdded(targ);
  if (noiseFactor <= 1.) return;  // Skip if not adding noise!

  typename BC::MVector noise;
#ifdef _OPENMP
#pragma omp critical(random)
#endif
  {
    noise = addedNoiseGenerator->sample(*gd);
  }
  targ.mom.m += noise;
  targ.cov.m += addedCovariance;
  return;
}

template<class CONFIG>
void
Prior<CONFIG>::prepare() {
  if (isPrepared)
    throw std::runtime_error("Two calls to Prior::prepare()");

  // Nothing to prepare for brute-force-integration Prior.
  
  // ??? This would be the place to precalculate template quantities

  isPrepared = true;
}

// The baseline routine that enters single copy of the template into sample vector.
// Optionally enters a flipped copy (of opposite parity).
template<class CONFIG>
void
Prior<CONFIG>::addSingleTemplate(const TemplateGalaxy<BC>& tmpl,
				 bool flip) {
  // Only one thread should alter the template pointer vector at a time.
#ifdef _OPENMP
#pragma omp critical(templates)
#endif
  {
    if (flip) {
      TemplateGalaxy<BC> tmpl2(tmpl);
      tmpl2.nda *=0.5;  // Split the density between 2 copies
      templatePtrs.push_back(new TemplateInfo<BC>(tmpl2));
      tmpl2.yflip();
      templatePtrs.push_back(new TemplateInfo<BC>(tmpl2));
    } else {      
      templatePtrs.push_back(new TemplateInfo<BC>(tmpl));
    }
  } // end omp single region
}


template <class CONFIG>
void
Prior<CONFIG>::addTemplate(const TemplateGalaxy<BC>& tmpl,
			   bool flip) {

#ifdef _OPENMP
#pragma omp critical(ndaTotal)
#endif
  {
    ndaTotal += tmpl.nda;
  }
  const int N_SELECT_ROTATIONS=6;  // Number of rotated copies
  // of template to use for detection probability calculation
  
  typedef TemplateGalaxy<BC> TG;
  if (isPrepared)
    throw std::runtime_error("Prior cannot addTemplate() after prepare()");

  // Determine suppression of this template from the factor
  // J * L_xy
  // Expression the J factor as a chisq:
  FP chisq = -2. * log(tmpl.jSuppression);
  // Position term, if we have one
  if (!BC::FixCenter) {
    // Get the center position in complex form
    auto z = tmpl.xyDeriv(TG::MX, TG::D0);
    // Calculate chisq contribution given that nominal covariance is diagonal in xy
    chisq += (z.real()*z.real()+z.imag()*z.imag()) /
      nominalCovariance.xy(BC::MX,BC::MX);
  }
  
  // Additional suppression due to flux being outside of selection bounds;
  // Approximating this by using the diagonal MF element of covariance
  auto f = real(tmpl.mDeriv(TG::MF, TG::D0));
  FP dflux = 0.;
  if (fluxMin>0.)
    dflux = std::max(dflux,fluxMin - f);
  if (fluxMax>0.)
    dflux = MAX(dflux, f - fluxMax);
  if (dflux>0.) 
    chisq += dflux*dflux / nominalCovariance.m(BC::MF,BC::MF);

  // Do not use this template if it is always outside cutoff
  if (chisq > sigmaCutoff*sigmaCutoff) return;

  // Now ready to add this template to the list
  // 
  if (selectionOnly && invariantCovariance) {
      // For invariant covariance matrix, we only need to tally the
      // selection Pqr once.  Do so for rotated copies of the template.
      TG trotate(tmpl);
      XYLike<BC> xyl(TargetGalaxy<BC>(typename BC::MVector(),nominalCovariance));
      trotate.nda /= N_SELECT_ROTATIONS; // Split density btwn rotated copies
      if (flip) trotate.nda /= 2.;       // Split between flipped copies
      double dtheta = 2.*PI / N_SELECT_ROTATIONS;

      for (int irot=0; irot<N_SELECT_ROTATIONS; irot++) {
	// Total selection probability is select term
	Pqr<BC> spqr = nominalSelector->detectPqr(trotate.realMDerivs());
	// times XY likelihood
	if (!BC::FixCenter) spqr *= xyl(trotate.realXYDerivs());
	// Times density
	spqr *= trotate.nda;
#ifdef _OPENMP
#pragma omp critical(templates)
#endif
	{
	  for (int i=0; i<BC::DSIZE; i++)
	    nominalSelectionPqr[i] += spqr[i];
	  ++selectionTemplateCounter;  // Keep track of templates used
	}
	if (flip) {
	  // Add flipped version
	  trotate.yflip();
	  spqr = nominalSelector->detectPqr(trotate.realMDerivs());
	  if (!BC::FixCenter) spqr *= xyl(trotate.realXYDerivs());
	  spqr *= trotate.nda;
#ifdef _OPENMP
#pragma omp critical(templates)
#endif
	  {
	    for (int i=0; i<BC::DSIZE; i++)
	      nominalSelectionPqr[i] += spqr[i];
	  }

	  trotate.yflip(); // Flip back
	}
	// Rotate object to next position
	trotate.rotate(dtheta);
      }
      // There is no need to even save the templates.

  } else if (selectionOnly) {
    // If we are only doing selection, we do not have to rotate the templates,
    // we'll just be building Selectors with rotated covariance matrices
    addSingleTemplate(tmpl, flip);

  } else {
    // Create rotations that sample the region around E2=0
    // Total amplitude of the template 2nd moments:
    auto z = tmpl.mDeriv(TG::ME,TG::D0);
    double absE=sqrt(z.real()*z.real() + z.imag()*z.imag());

    // Nominal covar is isotropic here, use sigma in E1
    double sigmaE = sqrt(nominalTotalCovariance.m(BC::M1,BC::M1));
    // Number of sigma of ME before total chisq goes over limit
    double maxESigma = sqrt(sigmaCutoff*sigmaCutoff - chisq);

    // Choose nominal rotation angle steps, up to PI/2.  
    double dbeta=PI/2.;
    if ( absE*dbeta > sigmaStep * sigmaE / 2.)
      dbeta = sigmaE * sigmaStep / (2.*absE);

    // Compute the maximum beta that we will want, up to +- PI/2, and resize dbeta to 
    // put integer steps in the range 
    int nSteps;
    double max_beta;
    if (maxESigma * sigmaE >= absE) {
      // Cover the full range of rotations, with at least 2 steps.
      max_beta = PI/2.;
      nSteps = static_cast<int> (ceil( 2*max_beta / dbeta));
      if (nSteps < 2) nSteps=2;
    } else {
      // Cover the +-max_beta range with integer number of steps:
      max_beta = asin(maxESigma*sigmaE / absE) / 2.;
      nSteps = static_cast<int> (ceil( 2*max_beta / dbeta));
      if (nSteps < 1) nSteps=1;
    }
    dbeta = 2*(max_beta) / nSteps;

    // compute the rotation angle in E plane that puts E2=0, E1>=0:
    double beta=-0.5*atan2(z.imag(), z.real());
    // Offset beta by random fraction of dbeta from the starting point

#ifdef _OPENMP
#pragma omp critical(random)
#endif
    beta += -max_beta + ud()*dbeta;

    // Make a copy of the template that we'll rotate
    TG tmpl2(tmpl);
    tmpl2.rotate(beta);
    
    // Disperse density among the rotated copies isotropically
    tmpl2.nda *= dbeta / (2.*PI);

    // Add a template at each rotation step:
    for (int iStep=0; iStep<nSteps; iStep++, beta+=dbeta) {
      addSingleTemplate(tmpl2, flip);
      tmpl2.rotate(dbeta);
    }
  }

  return;
}

/////////////////////////////////////////////////////////////////////////////////////////
// Routine to integrate over prior a target galaxy
/////////////////////////////////////////////////////////////////////////////////////////

template<class CONFIG>
Pqr<CONFIG> Prior<CONFIG>::getPqr(const TargetGalaxy<BC> &gal,
				  int& nTemplates,
				  int& nUnique) const {
  if (!isPrepared) 
    throw std::runtime_error("Called Prior::getPqr before prepare()");

  if (selectionOnly) {
    if (invariantCovariance) {
      // Selection probability is always the same!
      // But there is a potential type change here from double to float,
      // so copy explicitly:
      Pqr<BC> out;
      for (int i=0; i<out.size(); i++)
	out[i]=nominalSelectionPqr[i];
      return out;
    }
    // Get selection density.
    // Will create a small fleet of Selectors with target covariance
    // rotated through 2pi (instead of having made rotated copies of
    // the templates)
    const int N_SELECT_ROTATIONS=6;
    const double dbeta = 2.*PI / N_SELECT_ROTATIONS;
    TargetGalaxy<BC> galrot(gal);
    vector<const Selector<BC>*> vselect;
    vector<Pqr<BC>> vpqr(N_SELECT_ROTATIONS); // ??? Make these Pqr's double
    vector<XYLike<BC>> vxy;

    for (int i=0; i<N_SELECT_ROTATIONS; i++) {
      // Get the selection type appropriate to current usage.
      // We do not need to worry about added noise
      // to calculate the selection probability.
      if (fixedNoise) {
	vselect.push_back(new FixedNoiseSelector<BC>(fluxMin, fluxMax, galrot));
      } else {
	vselect.push_back(new Selector<BC>(fluxMin, fluxMax, galrot));
      }
      vxy.push_back(XYLike<BC>(galrot)); // Need an XYLike at each rotation too
      // Rotate covariance matrix for next step
      galrot.rotate(dbeta);
    }

    // ??? Could subsample the templates
    // ??? also potentially multithread here
    for ( auto tp : templatePtrs) {
      for (int j=0; j<N_SELECT_ROTATIONS; j++) {
	Pqr<BC> s = vselect[j]->detectPqr(tp->dm);
	FP p = s.getP();
	/**/if (s!=s) {
	  // Trap NaN's
	  cerr << "----Bad Pqr: " << s << endl;
	  cerr << " for rotation " << j << " moments:\n" << tp->getM() << endl;
	}
	s *= tp->nda;
	// Add factor for centroid likelihood:
	if (!BC::FixCenter)
	  s *= vxy[j](tp->dxy);
	p = s.getP();
	/**/ if (false) {
	  cerr << "detPQR: " << vselect[j]->detectPqr(tp->dm) << endl;
	  cerr << "nda: " << tp->nda << endl;
	  cerr << "xy: " << vxy[j](tp->dxy) << endl;
	}
	vpqr[j] += s;
      }
    }

    // Now sum up the pqr's after rotating each back to original orientation
    Pqr<BC> total;
    for (int i=0; i<N_SELECT_ROTATIONS; i++) {
      vpqr[i].rotate(-i*dbeta);
      total += vpqr[i];
      delete vselect[i];
    }
    total /= FP(N_SELECT_ROTATIONS);
    return total;
  }

  // From here out we're doing P(M,s).
  // Return negative probability if target flux is not selectable
  if (!nominalSelector->select(gal.mom)) {
    Pqr<BC> out;
    out[BC::P] = -1.;
    return out;
  }

  // We will rotate the input galaxy so it has E2 moment = 0 and E1 >=0. 
  // More exactly we are rotating the coordinate system, which leaves prior invariant.
  // Then at the end we rotate the Pqr result back to the original coordinate system.

  TargetGalaxy<BC> targ(gal);
  TargetGalaxy<BC> targAdded(gal);
  // Add any requisite additional noise:
  addNoiseTo(targAdded);

  // Find rotation angle that nulls E2:
  double beta = -0.5*atan2(targAdded.mom.m[BC::M2], targAdded.mom.m[BC::M1]);

  // Rotate galaxies both before and after adding noise, we might
  // need both covariance matrices
  targ.rotate(beta);
  targAdded.rotate(beta);

  // Our choice of selection function terms for the Pqr depends on
  // the type of measurement we are making:
  const Selector<BC>* s = chooseSelector(targ, targAdded);
  
  // Create an accumulator for this target, applying weights.
  PqrAccumulator<BC> accum(targAdded,
			   s,
			   sigmaCutoff,
			   true,
			   invariantCovariance);

  // Accumulate all templates in brute-force method
  for( auto tp : templatePtrs)
    accum.accumulate(tp);

  nTemplates = accum.nTemplates;
  nUnique = accum.nUnique();
  Pqr<BC> out; // copy double-valued accumulator into output
  for (int i=0; i<out.size(); i++)
    out[i]=accum.total[i];
  // Now un-rotate the result for this target galaxy
  out.rotate(-beta);

  delete s; // clean up

  return out;
}

template<class CONFIG>
const Selector<CONFIG>*
Prior<CONFIG>::chooseSelector(const TargetGalaxy<BC>& targ,
			      const TargetGalaxy<BC>& targAdded) const {
  // Note that we ignore the added noise if we're only doing selection probs:
  if (noiseFactor > 1. && !selectionOnly) {
    if (fixedNoise) {
      return new AddFixedNoiseSelector<BC>(fluxMin, 
					   fluxMax,
					   targ,
					   targAdded);
    } else {
      // Added noise, not fixed
      return new AddNoiseSelector<BC>(fluxMin, 
				      fluxMax,
				      targ,
				      targAdded);
    }
  } else {
    if (fixedNoise) {
      return new FixedNoiseSelector<BC>(fluxMin, 
					fluxMax,
					targ);
    } else {
      // Baseline case, no added noise, varies with center.
      return new Selector<BC>(fluxMin, 
			      fluxMax,
			      targ);
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////
// Instantiations:
/////////////////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE(...) \
  template class bfd::Prior<BfdConfig<__VA_ARGS__>>; 

#include "InstantiateMomentCases.h"
