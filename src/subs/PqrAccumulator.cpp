#include "PqrAccumulator.h"

using namespace bfd;

template<class CONFIG>
int
PqrAccumulator<CONFIG>::idCeiling = 200;

template <class CONFIG>
PqrAccumulator<CONFIG>::PqrAccumulator(const TargetGalaxy<BC>& targ,
				       const Selector<BC>* select_,
				       FP sigmaMax,
				       bool applyWeights_,
				       bool invariantCovariance_):
  total(),
  nTemplates(0),
  ml(targ, sigmaMax),
  xyl(targ),
  select(select_),
  applyWeights(applyWeights_),
  invariantCovariance(invariantCovariance_),
  stillCounting(true)
{
  // If the selection factor is independent of the template galaxy,
  // (and hence of lensing too) save the scalar value
  if (!select->probDependsOnG())
    selectionScalar = select->prob(targ.mom);
}


template <class CONFIG>
void
PqrAccumulator<CONFIG>::accumulate(const TemplateInfo<BC>* tmpl) {
  // This is the innermost loop of all the Sampled-Prior routines,
  // so we want it to be fast.  So I will be a little tricky with the linear algebra.

  // Get the chisq from the xy portion of the template to help decide quickly
  // whether to abort template
  FP chixy = 0.;
  if (!BC::FixCenter) {
    typename BC::XYVector xy(tmpl->dxy.col(BC::P));
    chixy = xyl.chisq(xy);
  }
  
  // Get the derivatives of the even-moment likelihood
  Pqr<BC> summand = ml(tmpl->dm, chixy);
  // If P<=0, no contribution here
  if (summand[BC::P] <= 0.)
    return;
  
  if (!BC::FixCenter) {
    // Multiply the XY likelihood
    summand *= xyl(tmpl->dxy);
  }
  
  if (select->probDependsOnG()) {
    // Selection term next
    summand *= select->probPqr(tmpl->dm);
  } else {
    // There is a fixed scalar selection factor
    summand *= selectionScalar;
  }
  
  if (applyWeights) summand *= tmpl->nda;

  for (int i=0; i<BC::DSIZE; i++)
    total[i] += summand[i]; // Accumulate! Might be mixed types, not overloaded by TMV
  
  // Increment counter of templates and add to set of used objects if we've not pegged already
  nTemplates++;
  if (stillCounting) {
    ids.insert(tmpl->id);
    if (ids.size()>=idCeiling) stillCounting = false;
  }
}

#define INSTANTIATE(...) \
  template class bfd::PqrAccumulator<BfdConfig<__VA_ARGS__>>;

#include "InstantiateMomentCases.h"
