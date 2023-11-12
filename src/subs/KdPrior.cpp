#include "KdPrior.h"
#include <cassert>
#include <algorithm>

// Get headers needed for Cholesky decomposition
#ifdef USE_TMV
#include "tmv/TMV_SymCHD.h"
#elif defined USE_EIGEN
#include "Eigen/Cholesky"
#endif

const int DEBUG=3;

using namespace bfd;

template<class Data>
int Node<Data>::maxElements=50;

template<class Data>
Node<Data>::Node(dataIter begin_, 
		 dataIter end_, 
		 SplitMethod split):
  mean(FP(0)),radius2(0.),
  maxMoment(FP(0)),minMoment(FP(0)),
  left(0), right(0),
  _begin(begin_),_end(end_),wTot(0.)
 {
  
  Assert(_end>_begin); // Not an empty set.  Can compare random-access iterators this way.

  // Determine bounds and mean of each moment
  mean.setZero();
  maxMoment = (*_begin)->getM();
  minMoment = (*_begin)->getM();

  for(auto i : *this) {
    mean+=i->getM();
    for(int idim=0;idim<DIM;++idim) {
      maxMoment[idim]=max(i->getM()[idim],maxMoment[idim]);
      minMoment[idim]=min(i->getM()[idim],minMoment[idim]);
    }
  }
  mean/= FP((_end-_begin));
  
#ifdef BALLTREE
  // Find point with maximum radius
  radius2 = 0.;
  Position dm;
  for(auto i : *this) {
    dm = i->getM()-mean;
    FP dist=dm.dot(dm);
    radius2=max(radius2,dist);
  }
#endif

  if( _end-_begin<= maxElements) {
    // If this is a leaf node, all we need to do is sum the element weights
    for(auto i : *this) {
      wTot+=i->nda;
    }
  }
  else {
    // Split the node along longest dimension
      
    // find the dimension with the greatest spread
    FP max_spread=0.;
    int splitAxis = -1;
    for(int idim=0;idim<DIM;++idim) {
      if( maxMoment[idim]-minMoment[idim] > max_spread) {
	splitAxis=idim;
	max_spread=maxMoment[idim]-minMoment[idim];
      }
    }

    if (splitAxis < 0) {
      // Get here if all dimensions have zero extent.  Make this a leaf
      // in this case of identical points.
      /**/cerr << "Node with zero extent at " << minMoment << endl;
      for(auto i : *this) {
	wTot+=i->nda;
      }
    } 
    else {
      // Split this node into two
      FP splitValue;
      if(split==MEAN) {
	splitValue=mean[splitAxis];
      } else if (split==MID) {
	splitValue=0.5*(minMoment[splitAxis] + maxMoment[splitAxis]);
      } else if (split==MEDIAN) {
	vector<FP> values;
	values.reserve(_end-_begin);
	for(auto i : *this) {
	  values.push_back(i->getM()[splitAxis]);
	}
	long midpt= values.size()/2;
	std::nth_element(values.begin(), values.begin()+midpt, values.end());
	splitValue = values[midpt];
      } else {
	throw std::runtime_error("Split method not implemented");
      }

      SplitCompareVal<Data> comp(splitAxis,splitValue);
      dataIter splitIter = std::partition(_begin, _end, comp);

      try {
	left=new Node<Data>(_begin,splitIter,split);
	right=new Node<Data>(splitIter,_end,split);
      }
      catch (std::bad_alloc) {
	throw std::runtime_error("Cannot allocate new node");
      }

      // Set up total weights:
      wTot = left->wTot + right->wTot;
    }
  }

}

template<class Data>
long Node<Data>::getNodeCount() const {
  if (left)
    return left->getNodeCount() + right->getNodeCount();
  else
    return 1L;
}

template<class Data>
void Node<Data>::minMax(const Position& m, FP& mind, FP& maxd) const
{
  mind=0,maxd=0;

#ifdef BALLTREE
  Position dm = m-mean;
  FP dist=dm.dot(dm);
  mind=MAX(FP(0),dist-radius2);
  maxd=dist+radius2;
#else
  // Find the closest and farthest point from each dimension and
  // add to squared distance

  for(int idim=0;idim<DIM;++idim) {
    FP low=minMoment[idim]-m[idim];
    FP high=m[idim]-maxMoment[idim];
    mind+=SQR(low+std::abs(low) + high+std::abs(high));
    maxd+=SQR(MAX(std::abs(low), std::abs(high)));
  }

  mind *= 0.25; 
#endif

}


template<class Data>
void Node<Data>::inOrOut(const Position& m, 
			 FP rInner2, FP rOuter2, 
			 std::vector<Node*>& in, NodeQueue& out,
			 double &sum)
{

  FP mind, maxd;
  minMax(m,mind,maxd);

  // If the minimum distance is greater than rInner then we put it
  // in the out pile
  if(mind > rInner2) {
    out.push(NodeWithDist(this,mind));
    // !! Take this out for now ???
    //sum+=std::exp(-mind/2.)*this->getWeightTotal();
    return;
  }

  // If the maximum distance from the points is less than rOuter
  // add the node to the inlist
  if( maxd<rOuter2) {
    in.push_back(this);
    return; 
  }

  if (left) {
    // Node spans inner/outer.  Descend tree
    left->inOrOut(m,rInner2, rOuter2, in, out,sum);
    right->inOrOut(m,rInner2, rOuter2, in, out,sum);
  } else {
    // A leaf node spanning the inner boundary goes on the inlist
    in.push_back(this);
  }
}
 
template<class Data>
Data* 
Node<Data>::sampleWeights(double cumulativeWeight) const
{
  if(!left) {
    // Leaf node.  Find element holding this cumulative weight.
    // Could build probability trees which might be faster here.

    dataIter index = _begin;
    while (cumulativeWeight > 0. && index!=_end) {
      index++;
      cumulativeWeight -= (*index)->nda;
    }
    Assert(index !=_end); // !! Warning here, it happens sometimes from roundoff?
    return *index;
  }
  else {
    // Pass the sample to children
    double wleft=left->getWeightTotal();
    if (cumulativeWeight < wleft) return left->sampleWeights(cumulativeWeight);
    else return right->sampleWeights(cumulativeWeight-wleft);
  }
}

template<class Data>
void
Node<Data>::sampleWeights(vector<double>::const_iterator wBegin,
			  vector<double>::const_iterator wEnd,
			  double weightOffset,
			  vector<Data*>& dataVector) const {
  if (wBegin==wEnd) return; // nothing to get from this node
  if (!left) {
    auto j = wBegin;
    for (dataIter index = _begin; index!=_end; ++index) {
      weightOffset += (*index)->nda;
      while (*j < weightOffset) {
	dataVector.push_back(*index);
	++j;
	if (j==wEnd) return;
      }
    }
    // Making it to the end of this loop is a failure (rounding error maybe?)
    // Instead of throwing exception, print a warning and return the last element

    cerr << "WARNING:  Node::SampleWeights at leaf with input weight "
	 << *j
	 << " but endpoint node weight "
	 << weightOffset
	 << endl;
    dataVector.push_back(*(_end-1));

  } else {
    // divide between left and right
    double wleft = left->getWeightTotal();
    auto wSplit=lower_bound(wBegin, wEnd, weightOffset+wleft);
    left->sampleWeights(wBegin, wSplit, weightOffset, dataVector);
    right->sampleWeights(wSplit, wEnd, weightOffset+wleft, dataVector);
  }
}
 
template<class Data>
Data* 
Node<Data>::sampleElements(long index) const
{
  // Since elements rooted to this node are gauranteed to be contiguous, we don't
  // even need to descend the tree to get an element:
  Assert(index>=0 && index < (_end-_begin));
  return *(_begin + index);
}
  
template<class Data>
void
Node<Data>::sampleElements(vector<long>::const_iterator elBegin,
			   vector<long>::const_iterator elEnd,
			   long elOffset,
			   vector<Data*>& dataVector) const {
  // Since elements rooted to this node are gauranteed to be contiguous, we don't
  // even need to descend the tree to get elements:
  for (auto i = elBegin;
       i != elEnd;
       ++i) {
    dataVector.push_back(*(_begin + (*i-elOffset)));
  }
}

template<class Data>
void
Node<Data>::getAllElements(std::vector<Data*> &vec)
{
  // No need to descend tree as all elements rooted here are between begin & end
  vec.insert(vec.end(), _begin, _end);
}

template<class CONFIG>
KDTreePrior<CONFIG>::KDTreePrior(FP fluxMin_,
				 FP fluxMax_,
				 const MomentCovariance<BC>& nominalCov,
				 ran::UniformDeviate<double>& ud_,
				 int nSamp_,
				 bool selectionOnly_,
				 FP noiseFactor_,
				 FP sigmaStep_,
				 FP sigmaCutoff_,
				 FP sigmaBuffer_,
				 bool invariantCovariance_,
				 bool fixedNoise_,
				 SplitMethod split_):
  Prior<CONFIG>(fluxMin_,fluxMax_,nominalCov,
		ud_, selectionOnly_, noiseFactor_,
		sigmaStep_, sigmaCutoff_,
		invariantCovariance_, fixedNoise_),
	    nSamp(nSamp_),
	    sigmaBuffer(sigmaBuffer_),
	    topNode(0),
	    split(split_),
	    sampleWeights(true)
{}

// Build the tree here
template <class CONFIG>
void
KDTreePrior<CONFIG>::prepare() {
  
  // Calling base class version will check & set the isPrepared flag
  Prior<BC>::prepare();

  // No need to build the tree if we are just doing selection probablities:
  if (selectionOnly)
    return;

  // Get Cholesky decomposition of the inverse nominal covariance matrix
  // If we are adding noise to the moments, take into account

  typename BC::MXYMatrix invcov(FP(0));
  invcov.subMatrix(0,BC::MSIZE,0,BC::MSIZE) = 
    Prior<BC>::nominalTotalCovariance.m.inverse();
  if (!BC::FixCenter)
    invcov.subMatrix(BC::MSIZE,BC::MXYSIZE,BC::MSIZE,BC::MXYSIZE) = 
      Prior<BC>::nominalTotalCovariance.xy.inverse();
  
  // Acquire lower triangular matrix of Cholesky decomposition
#ifdef USE_TMV
  {
    tmv::Matrix<FP> tmp(invcov);
    tmv::CH_Decompose(tmv::SymMatrixViewOf(tmp,tmv::Lower));
    A=tmp.lowerTri();
  }
#elif defined USE_EIGEN
  A = Eigen::LLT<typename BC::MXYMatrix::Base,Eigen::Lower>
    (invcov.template selfadjointView<Eigen::Lower>()).matrixL();
#endif

  // Multiply template moments by A to be in new basis
  typename BC::MXYVector allm;
  for(auto tptr : templatePtrs) {
    for(int i=0; i<BC::MSIZE; ++i)
      allm[i] = tptr->dm(i, BC::P);
    if (!BC::FixCenter) 
      for(int i=0; i<BC::XYSIZE; ++i)
	allm[i+BC::MSIZE] = tptr->dxy(i, BC::P);

    typename BC::MXYVector mTransformed = A.transpose()*allm;
    tptr->setM(mTransformed);
  }

  topNode=new Node<TemplateData>(templatePtrs.begin(),
				 templatePtrs.end(),
				 split);

  /**/cerr << "Number of nodes: " << topNode->getNodeCount() << endl;
  /**/cerr << "Number of templates: " << templatePtrs.size() << endl;
  /**/cerr << "Capacity of templates: " << templatePtrs.capacity() << endl;

  /**cerr << "Size of template: " << sizeof(MomentData) << endl;
  cerr << "Size of derivs: " << sizeof(MomentDerivs<UseMoments>) << endl;
  cerr << "Size of nodes: " << sizeof(Node<MomentData>) << endl;
  cerr << "Size of moments: " << sizeof(Moments<UseMoments>) << endl;
  **/
}

template<class CONFIG>
Pqr<CONFIG>
KDTreePrior<CONFIG>::getPqr(const TargetGalaxy<BC>& gal,
			    int& nTemplates,
			    int& nUnique) const
{
  if (!Prior<BC>::isPrepared) 
    throw std::runtime_error("Called KDTreePrior::getPqr2 before prepare()");

  // For selection, there is no subsampling from tree and base class call is good.
  if (selectionOnly)
    return Prior<BC>::getPqr(gal,nTemplates,nUnique);
    
  // Return negative probability if target flux is not selectable
  if (!Prior<BC>::nominalSelector->select(gal.mom)) {
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
  Prior<BC>::addNoiseTo(targAdded);

  // Find rotation angle that nulls E2:
  double beta = -0.5*atan2(targAdded.mom.m[BC::M2], targAdded.mom.m[BC::M1]);

  // Rotate galaxies both before and after adding noise, we might
  // need both covariance matrices
  targ.rotate(beta);
  targAdded.rotate(beta);

  //***********************************************
  // Now we'll subsample the template set using KdTree
  // to get a set to integrate, and a subsampling factor
  // to apply to the answer when done.
  //***********************************************

  // This holds pointers to the templates we will use
  std::vector<TemplateData*> useTemplates; 

#ifdef _OPENMP
#pragma omp critical(io)
  if (DEBUG>3)
    cerr << "#-- Thread " << omp_get_thread_num() << " collecting templates" << endl;
#endif

  // store the closest nodes ordered by distance to the current moment
  std::vector<NodePtr> nodes;
  NodeQueue outNodes;

  // Transform the target moments to the basis of identity nominalCovariance,
  // recalling targets must have zero XY moments
  typename BC::MXYVector allm(FP(0));
  allm.subVector(0,BC::MSIZE) = targAdded.mom.m;
  typename BC::MXYVector mSearch(A.transpose()*allm);

  // Get the nodes containing potentially useful templates
  FP inner2=SQR(Prior<BC>::sigmaCutoff);
  FP outer2=SQR(Prior<BC>::sigmaCutoff + sigmaBuffer);
  double sum=0.;
  topNode->inOrOut(mSearch,inner2,outer2,nodes,outNodes,sum);
  
  // Decide whether we will use all of the templates in these nodes,
  // or subsample
  bool subSample = nSamp>0;
  long nElements = 0L;
  for (int i=0; i<nodes.size(); i++)
    nElements += nodes[i]->getElementCount();

  //**/cerr << "nElements " << nElements << endl;
  
  // This factor multiplies our accumulated Pqr to normalize it to
  // the value of \sum {w_i P_i} that we would have obtained if we had
  // used all templates.
  double integralMultiplier = 1.;
  if (subSample) {
    // If we need more than this fraction of all elements, just get them all:
    const double MaxFractionToSample = 0.5;

    double fractionOfElements = nSamp;
    fractionOfElements /= nElements;
    if (fractionOfElements > MaxFractionToSample) {
      subSample = false;
    }
  }

  if (subSample) {
    // Select a random subset of the templates in our nodes
    // Get all the random deviates in one swoop to avoid too many thread collisions
    useTemplates.reserve(nSamp);
    vector<double> deviates(nSamp);
#ifdef _OPENMP
#pragma omp critical(random)
#endif
    {
      for (int i=0; i<nSamp; i++) deviates[i] = ud();
    }
    
    std::sort(deviates.begin(), deviates.end());

    if (sampleWeights) {
      // Get templates with likelihood proportional to weight
      double wTot = 0.;
      for (int i=0; i<nodes.size(); i++) {
	wTot += nodes[i]->getWeightTotal();
      }
      for (long i=0; i<deviates.size(); i++) {
	deviates[i] *= wTot;
      }      

      integralMultiplier = wTot / nSamp;

      vector<double>::iterator wBegin = deviates.begin();
      vector<double>::iterator wEnd = deviates.end();

      double weightOffset = 0.;
      for (int inode=0; inode<nodes.size(); inode++) {
	// Split deviate array at total 
	double nodeWeight = nodes[inode]->getWeightTotal();
	vector<double>::iterator wSplit = std::lower_bound(wBegin, wEnd,
							   weightOffset + nodeWeight);
	// Collect the templates that are rooted at this node:
	nodes[inode]->sampleWeights(wBegin, wSplit, weightOffset,useTemplates);

	weightOffset += nodeWeight;
	wBegin = wSplit;
      }
    } else {
      // Select templates with equal probability

      // Make a vector of element indices
      nElements = 0L;
      for (int i=0; i<nodes.size(); i++) {
	 nElements += nodes[i]->getElementCount();
      }

      vector<long> vCounts(deviates.size());
      for (long i=0; i<deviates.size(); i++)
	vCounts[i] = static_cast<long> (floor(deviates[i]*nElements));

      integralMultiplier = nElements / ( (double) nSamp);

      // Partition the samples among the "in" nodes
      vector<long>::iterator elBegin = vCounts.begin();
      vector<long>::iterator elEnd = vCounts.end();
      long elOffset=0L;
      for (int inode=0; inode<nodes.size(); inode++) {
	// Split deviate array at total 
	long nodeCount = nodes[inode]->getElementCount();
	vector<long>::iterator elSplit = std::lower_bound(elBegin, elEnd,
							  nodeCount+elOffset);
	// Collect the templates that are rooted at this node:
	nodes[inode]->sampleElements(elBegin, elSplit, elOffset, useTemplates);

	elOffset += nodeCount;
	elBegin = elSplit;
      }
    }
  } else {
    // Not subsampling; use all of the templates from "in" nodes
    for (int i=0; i<nodes.size(); i++)
      nodes[i]->getAllElements(useTemplates);
  } // Done selecting our template set.

  // If we are using a weighted-probability subset of the templates, do NOT
  // subsequently multiply by weight inside the accumulator:
  bool applyWeights = !(subSample && sampleWeights);

  //***********************************************
  // Next do the sum over the sampled template galaxies.
  //***********************************************

  // Our choice of selection function terms for the Pqr depends on
  // the type of measurement we are making:
  const Selector<BC>* s = Prior<BC>::chooseSelector(targ, targAdded);
  
#ifdef _OPENMP
#pragma omp critical(io)
  if (DEBUG>3)
    cerr << "#-- Thread " << omp_get_thread_num() << " starting " << useTemplates.size() << " templates" << endl;
#endif

  // Create an accumulator for this target, applying weights.
  PqrAccumulator<BC> accum(targAdded,
			   s,
			   sigmaCutoff,
			   applyWeights,
			   invariantCovariance);

  // Accumulate all templates in brute-force method
  for( auto tp : useTemplates) {
    // Skip if it's outside the inner radius for the tree using nominal covariance
    typename BC::MXYVector diff=tp->getM()-mSearch;
    if(diff.dot(diff) > inner2) continue;
    accum.accumulate(tp);
  }

#ifdef _OPENMP
#pragma omp critical(io)
  if (DEBUG>3)
    cerr << "#-- Thread " << omp_get_thread_num() << " got p=" << accum.total[BC::P] << endl;
#endif

  // Block below is for debugging the negative probabilities. Can be removed?
#ifdef _OPENMP
#pragma omp critical(io)
#endif
  /**/ if (accum.total[BC::P]<=0) {
    cerr << "Zero or negative probability galaxy " << endl;
    cerr << "Target ID " << targ.id << endl;
    cerr << "Raw moments:\n " << targ.mom.m << endl;
    cerr << "Rotated moments:\n " << targAdded.mom.m << endl;
    cerr << nodes.size() << " active nodes " << nElements << " elements ";
    cerr << accum.nTemplates << " used "
	 << accum.nUnique() << " unique." << endl;
    cerr << "pqr:\n" << accum.total << endl;
    cerr << "...Hunting for negative-prob contributions..." << endl;
    for( auto tp : useTemplates) {
      // Skip if it's outside the inner radius for the tree using nominal covariance
      typename BC::MXYVector diff=tp->getM()-mSearch;
      if(diff.dot(diff) > inner2) continue;
      accum.total.setZero(); 
      accum.accumulate(tp);
      if (accum.total[BC::P]<0.) {
	cerr << "....Negative prob for template ID " << tp->id << endl;
	cerr << "....Moments:\n" << tp->getM() << endl;
	cerr << "....pqr:\n" << accum.total << endl;
      }
    }
  }

  /**/

  nTemplates = accum.nTemplates;
  nUnique = accum.nUnique();
  Pqr<BC> out; // Copy double-valued accumulator to output
  for (int i=0; i<out.size(); i++)
    out[i] = accum.total[i]; 
  // Now un-rotate the result for this target galaxy
  out.rotate(-beta);

  // Boost summed probabilities to compensate for subsampling:
  out *= FP(integralMultiplier);

  delete s; // clean up

  return out;
}

///////////////////////////////////////////////////////////////
// 
// Instantiations of the templates
//
///////////////////////////////////////////////////////////////

#define INSTANTIATE(...) \
  template class bfd::Node<TemplateInfo<BfdConfig<__VA_ARGS__>>>;	\
  template class bfd::KDTreePrior<BfdConfig<__VA_ARGS__>>;

#include "InstantiateMomentCases.h"
