#include "KdPrior.h"
#include "Random.h"
#include <vector>
#include <ctime>
#include <boost/program_options.hpp>
#include "Moments.h"
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/sum.hpp>
#include <boost/accumulators/statistics/variance.hpp>

namespace bacc = boost::accumulators;
typedef bacc::accumulator_set < double, bacc::stats 
				<bacc::tag::mean, bacc::tag::min,
				 bacc::tag::max, bacc::tag::variance, 
				 bacc::tag::median, bacc::tag::sum, 
				 bacc::tag::count> > TimeAccumulator;

std::ostream &operator<<(std::ostream &out, const TimeAccumulator &acc) {

    out << "count: " << bacc::count(acc) << "\n";
    out << "min: " << bacc::min(acc) << "\n";
    out << "max: " << bacc::max(acc) << "\n";
    out << "mean: " << bacc::mean(acc) << "\n";
    out << "variance: " << bacc::variance(acc) << "\n";
    out << "median: " << bacc::median(acc) << "\n";
    out << "sum: " << bacc::sum(acc);
    return out;
}

#include <boost/timer/timer.hpp>
#include <boost/chrono/chrono.hpp>

using boost::timer::cpu_timer;
using boost::timer::cpu_times;
typedef boost::chrono::duration<double> sec; 
namespace po=boost::program_options;
using namespace ran;
using namespace bfd;
using namespace std;

int main(int argc,char *argv[])
{

  int ntree=100000;
  int ntest=1000;
  const int m=3;
  double max=6;
  double maxr=4;
  double minr=3;
  int nSamp=1000;
  int maxLeaf=10;
  

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "print this message")
    ("ntree",po::value<int>(&ntree)->default_value(100000),
     "Number of tree entries")
    ("ntest",po::value<int>(&ntest)->default_value(1000),
     "Number of test entries")
    ("max",po::value<double>(&max)->default_value(3),
     "Maximum for bounds")
    ("minr",po::value<double>(&minr)->default_value(2),
     "Minimum r for search")
    ("maxr",po::value<double>(&maxr)->default_value(3),
     "Maximum r for search")
    ("nsamp",po::value<int>(&nSamp)->default_value(1000),
     "Number of samples to get")
    ("maxleaf",po::value<int>(&maxLeaf)->default_value(-1),
     "Maximum number of leaf elements")
  ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if(vm.count("help")) {
    cout<< desc <<endl;
    return 1;
  }
  
  if(maxLeaf>0) Node<MomentInfo<12>,m>::setMaxElements(maxLeaf);

  cpu_timer cpu;
  
  std::vector<MomentInfo<12> > vec;  
  UniformDeviate u;
  for(int i=0;i<ntree;++i) {
    
    DVector p(m,0);
    for(int j=0;j<m;++j) { 
      p(j)=u*max;
    }
    Moments<12> mm(p);
    MomentDerivs<12> mom(mm);
    vec.push_back(MomentInfo<12>(mom));

  }
    
    
  SplitMethod sm=MEAN;
  cpu.start();
  TreeNode<MomentInfo<12>,m> node(vec,0,ntree,sm,100*SQR(maxr));
  node.calcElements();
  sec tseconds = boost::chrono::nanoseconds(cpu.elapsed().wall);
  
  cout<<"Tree built in "<<tseconds.count()<<" seconds"<<endl;  

  TimeAccumulator time_acc;
  for(int itest=0;itest<ntest;++itest) {
    cpu.start();

    // create random point
    DVector p(m);
    for(int j=0;j<m;++j) { 
      p(j)=u*max;
    }
    Moments<12> mm(p);
    MomentDerivs<12> mom(mm);
    
    std::vector<Node<MomentInfo<12>,m>*> nodes;
    Node<MomentInfo<12>,m >::NodeQueue out;
    double sum=0;
    node.inOrOut(mom,SQR(minr),SQR(maxr),nodes,out,sum);
    std::vector<int> nElements(nodes.size(),0);
    int ntot=0;
    //cout<<nodes.size()<<" "<<out.size()<<" "<<sum/ntree<<" "<<node.getElements()<<endl;

    if(nodes.size()==0) {
      cout<<"Did not find any objects within radius "<<p<<endl;
      continue;
    }

    for(int i=0;i<nodes.size();++i) {
    
      nElements[i]=ntot+nodes[i]->getElements();
      ntot+=nodes[i]->getElements();
    }

    std::vector<MomentInfo<12>*> use(nSamp);
    for(int isamp=0;isamp<nSamp;++isamp) {
      
      // random integer to be used to select which node will be
      // sampled
      double rand=u()*ntot;
      
      // find out the first element in the array that is less than rand
      std::vector<int>::iterator iter;
      iter=upper_bound(nElements.begin(),nElements.end(),rand);
      int nodeIndex=iter-nElements.begin();

      // the number of elements before this node
      int nbefore(0);
      if(nodeIndex>0) nbefore=nElements[nodeIndex-1];

      MomentInfo<12>* temp=nodes[nodeIndex]->sample(rand-nbefore);
      use[isamp]=temp;

    }
    
    
    tseconds = boost::chrono::nanoseconds(cpu.elapsed().wall);
    time_acc(tseconds.count());
  }
  cout<<time_acc<<endl;
  
}



 
