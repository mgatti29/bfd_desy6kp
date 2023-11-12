// An implementation of the Prior class that keep the SampledPrior templates
// in a KdTree

/**********
Things to do:
* Check the neglected probability sum compared to included part
* Specify a cutoff on the exact sigma when feeding PqrAccumulator??
* Check on bias calculations with weights in use, different samplings.
* Possible repeated sampling when bias estimate is too large.
**********/

#ifndef KDPRIOR_H
#define KDPRIOR_H
#include "Std.h"
#include <vector>
#include <list>
#include <map>
#include <queue> 
#include "Prior.h"

namespace bfd {

  const double defaultValue=9999;

  // Which split method to use for tree nodes:
  // MEAN is the average 
  // MEDIAN splits the number of objects into two
  // MID splits according to the middle of the bounds 
  enum SplitMethod { MEAN, MEDIAN, MID};

  
  // Helper functions to sort data based on different indexes
  // Assume that we will be manipulating pointers to a structure
  template<class T>
  struct SplitCompare
  {
    SplitCompare(int _split) : split(_split) {}
    bool operator()(const T* a, const T* b) const
    { 
      return a->getM()[split] < b->getM()[split]; 
    }
    int split;
  };
  
  template<class T>
  struct SplitCompareVal
  {
    SplitCompareVal(int _split,double _val) :
      split(_split),val(_val) {}
    bool operator()(const T* a) const
    { 
      return a->getM()(split) < val; 
    }
    
    int split;
    double val;
  };
  


  // KDTree construction
  // Data here is an object that includes a typedef for the  
  // MomentIndices class
  // as a typedef:
  //
  //  typedef MomentIndices<UseMoments> MI;
  //
  // This stores the dimension of the tree and some of the data structures.
  // We could make this more generic in the future if we need to extend
  // this class.
  //
  // Node is not responsible for destruction of data members.
  // The manipulations all occur on a vector of Data* pointers.
  //
  // Can #define BALLTREE to use this instead of default boxes.

  template<class Data>
  class Node {
    
  public:
    typedef typename vector<Data*>::iterator dataIter;
    typedef typename Data::MXYVector Position;  // Position in space being divided
    typedef typename Data::FP  FP;              // Data type of position elements
    // (using double precision for all cumulative quantities)
    static const int DIM=Data::BC::MXYSIZE;    

    // Need access to values only on construction.
    // Data values belonging to a Node are assumed to be contiguous in
    // a large array, so we specify just ptrs to the first and (one after) last
    // belonging to this node.  Children will split this range.
    Node(dataIter _begin, 
	 dataIter _end, 
	 SplitMethod split);

    // Destructor deletes children but not the members
    ~Node() {
      if (left) delete left;
      if (right) delete right;
    }

    // structure to use in priority queue, includes distance from target
    // which is used to maintain priorities.
    struct NodeWithDist
    {
      NodeWithDist(Node *_pos,double _dist):pos(_pos),dist(_dist) {}
      NodeWithDist():pos(0),dist(0.) {}
      Node* pos;
      FP dist;
      bool operator<(const NodeWithDist& rhs) const {return dist<rhs.dist;}
    };

    typedef std::priority_queue<NodeWithDist,std::vector<NodeWithDist>> NodeQueue;

    // Access the total elements, weight, or nodes rooted at this node:
    long getElementCount() const {return _end - _begin;}
    double getWeightTotal() const {return wTot;}
    long getNodeCount() const;
    
    // Sample a data point from among all those rooted at this Node.
    // First version takes as input the (random) integer index of the one to get,
    //  will throw exception if randomIndex<0 or randomIndex>=getElementCount().
    Data* sampleElements(long index) const;

    // In this version the iterators point to a range of sorted, random element numbers.
    // Data* pointers corresponding to these (random) element numbers
    //   are pushed onto the dataVector.
    // elOffset is subtracted from each input element number.
    void sampleElements(vector<long>::const_iterator elBegin,
			vector<long>::const_iterator elEnd,
			long elOffset,
			vector<Data*>& dataVector) const;

    // Select a random Data element with probability according to weight.
    // Input is (random) double giving point in cumulative weight function to select.
    // Exception unless 0 <= cumulativeWeight < getWeightTotal().
    Data* sampleWeights(double cumulativeWeight) const;

    // This time add to dataVector a Data element selected by each cumulative weight
    // in the range of the begin and end iterators.  wOffset is subtracted from each
    // input weight.
    void sampleWeights(vector<double>::const_iterator wBegin,
		       vector<double>::const_iterator wEnd,
		       double wOffset,
		       vector<Data*>& dataVector) const;


    // Build list of nodes holding all points within rinner of m.
    // First, nodes fully exterior to rinner are added to the "out" queue, ordered
    //   by their minimal distance to m.
    // Then remaining nodes fully interior to router are added to the "in" vector.
    // Remaining nodes are added to the "in" list if they are leaves, otherwise
    //   we descend the tree and consider the daughter nodes separately.
    // An upper limit to the weighted likelihood of all excluded nodes is added to sum.
    void inOrOut(const Position& m, 
		 FP rInner2, FP rOuter2, 
		 std::vector<Node*>& in, NodeQueue& out,
		 double &sum);  
    

    // get maximum/minimum squared distance from element
    // If point is inside circumscribed box of node members,
    // the minimum distance will be zero.
    void minMax(const Position& m, FP& mind, FP& maxd) const;

    // Append all points rooted to this node to the vector
    void getAllElements( std::vector<Data*> &vec);

    static void setMaxElements(int n) {maxElements=n;}
    
    EIGEN_NEW
      
  private:


    // centroid of node - used for BALLTREE or if split = MEAN
    Position mean;
    // maximum radius^2 of all objects on the node
    double radius2;

    // These arrays hold the minimum and maximum values of moments of objects
    // rooted at this node.
    Position minMoment;
    Position maxMoment;

            
    // Pointers to children
    Node* left;
    Node* right;

    // Total weight in this node
    double wTot;

    // Range of data elements contained in this node.
    // Gauranteed that all children of this node are in this contiguous range.
    dataIter _begin;
    dataIter _end;

    // These functions allow C++-11 for loop syntax:
    dataIter begin() const {return _begin;}
    dataIter end() const {return _end;}
    
    // This will be the same for all members of the tree
    static int maxElements;
  };


  ////////////////////////////////////////////////////////////////////////////////
  // Now an implementation of the Prior class that keep the SampledPrior templates
  // in a KdTree
  ////////////////////////////////////////////////////////////////////////////////

  template<class CONFIG>
  class KDTreePrior: public Prior<CONFIG> {

  public:
    typedef CONFIG BC;
    typedef typename BC::FP FP;
    typedef TemplateInfo<BC>  TemplateData;
    typedef Node<TemplateData>* NodePtr;
    typedef typename Node<TemplateData>::NodeQueue NodeQueue;

    // nSamp is the target number of templates to subsample.  nSamp<=0 bypasses subsampling.
    // sigmaBuffer is "slack" for keeping/tossing tree nodes.
    // split_ is method for dividing nodes.
    KDTreePrior(FP fluxMin_,
		FP fluxMax_,
		const MomentCovariance<BC>& nominalCov,
		ran::UniformDeviate<double>& ud_,
		int nSamp_,
		bool selectionOnly_ = false,
		FP noiseFactor_ = 1.,
		FP sigmaStep_ = 1.,
		FP sigmaCutoff_ = 6.5,
		FP sigmaBuffer_ = 1.,
		bool invariantCovariance_ = false,
		bool fixedNoise_ = false,
		SplitMethod split_=MEAN);  // MID better?

    virtual ~KDTreePrior() {
      if (topNode) delete topNode;
    }
    
    // Have to declare protected data members from templated base class explicitly
    using Prior<BC>::ud;
    using Prior<BC>::selectionOnly;
    using Prior<BC>::templatePtrs;
    using Prior<BC>::sigmaCutoff;
    using Prior<BC>::invariantCovariance;

    // Override these methods of the base class:

    // build tree and calculate number of objects/node
    virtual void prepare();

    // Do the calculations
    virtual Pqr<BC> getPqr(const TargetGalaxy<BC>& gal,
			   int& nTemplates,
			   int& nUnique) const;
      
    // With this as true (default), the sampling probability is according to
    // template weight, so weights do not need to be reapplied.
    // When false, all templates have equal chance of being sampled.
    void setSampleWeights(bool b) {sampleWeights=b;}

    EIGEN_NEW
      
  protected:

    // How many samples do we want
    int nSamp;
    
    // How many sigma between the inner and outer radii for tree?
    // (Inner radius will be the sigmaCutoff value in the SampledPrior class)
    FP sigmaBuffer;

    NodePtr topNode;
    SplitMethod split;

    // Diagonaliztion of inverse covariance matrix
    // into C^-1=A^T*A
    typename BC::MXYMatrix A;
    
    // sample by weights and do not weight likelihoods (vs sampling all templates equally)
    bool sampleWeights;
  };

} // namespace bfd

#endif
