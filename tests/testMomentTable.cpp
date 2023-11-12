// Test reading and writing of targets and templates from FITS files.
#include <iostream>
#include "Pqr.h"
#include "BfdConfig.h"
#include "MomentTable.h"
#include "FitsTable.h"
#include "FitsImage.h"
#include "testSubs.h"

using namespace bfd;
using namespace test;


const bool FIX_CENTER=false;
const bool USE_CONC = true;
// Should really try this test too:
//const bool USE_CONC = false; 
const bool USE_MAG = true;  // Use everything to exercise all code
const int N_COLORS=0;
const bool USE_FLOAT=true;

typedef BfdConfig<FIX_CENTER, USE_CONC, USE_MAG, N_COLORS, USE_FLOAT> BC;

Moment<BC> array2Moments(double d[]) {
  // Convert python-ordered moments as double array into C++ Moment structure
  Moment<BC> out;
  out.m[BC::MF] = d[TemplateTable<BC>::M0];
  out.m[BC::MR] = d[TemplateTable<BC>::MR];
  out.m[BC::M1] = d[TemplateTable<BC>::M1];
  out.m[BC::M2] = d[TemplateTable<BC>::M2];
  if (BC::UseConc) out.m[BC::MC] = d[TemplateTable<BC>::MC];
  if (!BC::FixCenter) {
    out.xy[BC::MX] = d[TemplateTable<BC>::MX];
    out.xy[BC::MY] = d[TemplateTable<BC>::MY];
  }
  return out;
}
  
bool compareDerivs(img::FTable& tab, TemplateGalaxy<BC>& tmpl,
		   string colname, int matrixCol,
		   LONGLONG index) {
  // Do comparison of input table real-valued derivatives and those
  // coming back out of the TemplateGalaxy
  Moment<BC> oldm;
  Moment<BC> newm;
  
  // Maps from C++ indices to the Python indices
  int mMap[BC::MSIZE];
  mMap[BC::MF] = TemplateTable<BC>::M0;
  mMap[BC::MR] = TemplateTable<BC>::MR;
  mMap[BC::M1] = TemplateTable<BC>::M1;
  mMap[BC::M2] = TemplateTable<BC>::M2;
  if (BC::UseConc) mMap[BC::MC] = TemplateTable<BC>::MC;

  int xyMap[BC::XYSIZE];
  if (!BC::FixCenter) {
    xyMap[BC::MX] = TemplateTable<BC>::MX;
    xyMap[BC::MY] = TemplateTable<BC>::MY;
  }

  auto md = tmpl.realMDerivs();
  auto xyd = tmpl.realXYDerivs();
  vector<float> vf;
  tab.readCell(vf, colname, index);
  for (int i=0; i<BC::MSIZE; i++) {
    oldm.m[i]=vf[mMap[i]];
    newm.m[i] = md(i,matrixCol);
  }
  if (!BC::FixCenter) {
    for (int i=0; i<BC::XYSIZE; i++) {
      oldm.xy[i] = vf[xyMap[i]];
      newm.xy[i] = xyd(i,matrixCol);
    }
  }
  cout << colname << endl;
  return test::compare(oldm,newm,"Old","New");
}

// Same comparison but compare table entries to a given array
bool compareDerivs2(img::FTable& tab, double ref[],
		    string colname, LONGLONG index) {
  // Do comparison of input table real-valued derivatives and those
  // coming back out of the TemplateGalaxy
  DVector tabm(TemplateTable<BC>::NM,0.);
  DVector refm(TemplateTable<BC>::NM,0.);
  
  vector<float> vf;
  tab.readCell(vf, colname, index);
  for (int i=0; i<TemplateTable<BC>::NM; i++) {
    tabm[i] = vf[i];
    refm[i] = ref[i];
  }
  cout << colname << endl;
  return test::compare(tabm,refm,"Read","Reference");
}

  
int
main(int argc,
     char *argv[])
{
  const string usage =
    "testMomentTable: see if TargetTable and TemplateTable reading/writing work\n"
    "Usage: testMomentTable [outfile]\n"
    " outfile: name at which to save altered targets [junk.fits]\n"  
    "Will access testdata/test_*table.fits files for input";
  if (argc > 2) {
    cerr << usage << endl;
    exit(1);
  }
  string targetOut = argc>1 ? argv[1] : "junk.fits";
  string targetIn = "testdata/test_target_table.fits";
  string targetCovIn = "testdata/test_target_covtable.fits";
  string templateIn = "testdata/test_template_table.fits";
  if (argc<=1) {
    cerr << usage << endl;
    cerr << "Proceeding with default arguments." << endl;
  }

  bool failure = false;

  // First check reading, altering, and re-writing of template table
  try {
    // Right answers for the target tables
    long index = 6;  // Look at object #6...
    long int id=6;
    double _m[] = {1379.01843262,   415.837677  ,   -18.20459366,   -15.2463274 ,  276.76168823, 0., 0.};
    auto mom = array2Moments(_m);
    double x = 0.25077206, y=0.51873648;
    double cove[] = {1.20076914e+04,   7.35737305e+03,  -6.46303594e-03,
		     9.80464478e+01,   7.03016748e+03,   7.03016748e+03,
		     -8.42487533e-03,   1.12557304e+02,   8.44895410e+03,
		     3.51509229e+03,   4.18858649e-03,  -1.11707393e-02,
		     3.51507495e+03,   1.51795090e+02,   1.17839355e+04};
    double covo[] = {3678.68334961,    49.02322388,  3678.68994141};

    // Do two iterations, first with a target table with a fixed common cov matrix,
    // then again with one having per-galaxy cov matrices.
    for (int itab = 0; itab < 2; itab++) {
      bool covFixed = itab==0;
      TargetTable<BC> tab1( covFixed ? targetIn : targetCovIn);
      cout << "---------- Testing Fixed-covariance target table ----------" << endl;
      if (covFixed) {
	if (!tab1.hasSharedCov()) {
	  cout << "========>FAILURE: hasSharedCov() returned false" << endl;
	  failure = true;
	}
      }	else {
	if (tab1.hasSharedCov()) {
	  cout << "========>FAILURE: hasSharedCov() returned true" << endl;
	  failure = true;
	}
      }
      try {
	auto cov = tab1.getNominalCov(0);
      } catch (std::runtime_error &e) {
	cout << "========>FAILURE: getNominalCov() throws exception" << endl;
	failure = true;
      }

      auto targ = tab1.getTarget(index);
      cout << "---Moment vector" << endl;
      failure = compare(targ.mom, mom, "Read", "Reference") || failure;
      cout << "---Id: " << endl;
      if (targ.id != id) {
	cout << "========>FAILURE: ID read: " << targ.id << " Reference " << id << endl;
	failure = true;
      }
      cout << "---X position: " << endl;
      failure = compare(targ.position[0], x, "Read", "Reference") || failure;
      cout << "---Y position: " << endl;
      failure = compare(targ.position[1], y, "Read", "Reference") || failure;

      // Compare covariance matrices element by element, only report errors
      cout << "---Covariance:" << endl;
      double covTol = 1e-4;
      // Maps from Python indices to the C++ indices - -1 if not used in C++
      int mMap[TargetTable<BC>::NM];
      mMap[TemplateTable<BC>::M0] = BC::MF;
      mMap[TemplateTable<BC>::MR] = BC::MR;
      mMap[TemplateTable<BC>::M1] = BC::M1;
      mMap[TemplateTable<BC>::M2] = BC::M2;
      mMap[TemplateTable<BC>::MC] = BC::UseConc ? BC::MC : -1;
      double *ref = cove;
      // Compare cov elements to the listing of upper triangle
      for (int i=0; i<TargetTable<BC>::NM; i++) {
	int ii = mMap[i];
	for (int j=i; j<TargetTable<BC>::NM; j++, ++ref) {
	  int jj = mMap[j];
	  if (ii>=0 && jj>=0 && abs(targ.cov.m(ii,jj)-*ref) > covTol) {
	    cout << "========>FAILURE: MCov mismatch at " << ii << "," << jj
		 << " Read " << targ.cov.m(ii,jj) << " Reference " << *ref << endl;
	    failure = true;
	  }
	  if (ii>=0 && jj>=0 && abs(targ.cov.m(jj,ii)-*ref) > covTol) {
	    cout << "========>FAILURE: MCov mismatch at " << jj << "," << ii
		 << " Read " << targ.cov.m(jj,ii) << " Reference " << *ref << endl;
	    failure = true;
	  }
	}
      }

      // XY part of covariance
      if (!BC::FixCenter) {
	if (abs(targ.cov.xy(0,0)-covo[0]) > covTol) {
	    cout << "========>FAILURE: XYCov mismatch at 0,0 "
		 << " Read " << targ.cov.xy(0,0) << " Reference " << covo[0] << endl;
	    failure = true;
	}
	if (abs(targ.cov.xy(0,1)-covo[1]) > covTol) {
	    cout << "========>FAILURE: XYCov mismatch at 0,1 "
		 << " Read " << targ.cov.xy(0,1) << " Reference " << covo[1] << endl;
	    failure = true;
	}
	if (abs(targ.cov.xy(1,0)-covo[1]) > covTol) {
	    cout << "========>FAILURE: XYCov mismatch at 1,0 "
		 << " Read " << targ.cov.xy(1,0) << " Reference " << covo[1] << endl;
	    failure = true;
	}
	if (abs(targ.cov.xy(1,1)-covo[2]) > covTol) {
	    cout << "========>FAILURE: XYCov mismatch at 1,1 "
		 << " Read " << targ.cov.xy(1,1) << " Reference " << covo[2] << endl;
	    failure = true;
	}
      }
      
      // Make a Pqr entry
      Pqr<BC> pqr;
      for (int i=0; i<pqr.size(); i++) pqr[i] = i;
      tab1.setPqr(pqr, index);
      cout << "---setPqr" << endl;
      failure = compare(tab1.getPqr(index), pqr, "Retrieved", "Reference") || failure;
      
      // Now double it
      tab1.addPqr(pqr, index);
      pqr += pqr;
      cout << "---addPqr" << endl;
      failure = compare(tab1.getPqr(index), pqr, "Retrieved", "Reference") || failure;

      // Now save the altered file
      /**
      tab1.save(targetOut);

      // Check the saved file's primary extension is null if it should be
      {
	img::FitsImage<double> fi(targetOut,FITS::ReadOnly,0);
	if (covFixed && !fi.getBounds()) {
	  cout << "========>FAILURE: saved table returned undefined primary image" << endl;
	  failure = true;
	} else if (!covFixed && fi.getBounds()) {
	  cout << "========>FAILURE: saved table returned non-null primary image" << endl;
	  failure = true;
	}
      }
	  
      // Reopen and check
      TargetTable<BC> tab2(targetOut);
      
      targ = tab2.getTarget(index);
      cout << "---Saved and retrieved moment:" << endl;
      failure = compare(targ.mom, mom, "Retrieved", "Reference") || failure;
      cout << "---Saved and retrieved pqr:" << endl;
      failure = compare(tab2.getPqr(index), pqr, "Retrieved", "Reference") || failure;
      cout << endl;
      **/
    }

  } catch (std::runtime_error& m) {
    cout << "FAILURE - exception: " << m.what() << endl;
    failure = true;
  }

  // Now tests of the TemplateTable
  try {
    
    // Right answer for the template table
    long int index = 822;
    long int id = 5;
    double nda = 0.00040609873;
    double jSuppression = 0.81609035;
    double moments[] = {1273.90026855,546.4085083,-28.84632683,-44.99890518,459.49932861,153.22192383,106.27992249};
    double moments_dg1[] = {-77.79471588,-34.37460709,-360.97012329,6.76122856,-25.668396,50.42781067,-73.05908203};
    double moments_dg2[] = {-136.20785522,-93.22753143,6.76122856,-359.11959839,-100.17448425,33.69616699,52.91603088};
    double moments_dmu[] = {1369.19189453,372.72729492,-34.37460709,-93.22753143,232.12353516,256.56576538,213.03517151};
    double moments_dg1_dg1[] = {-960.00463867,-394.19128418,98.69477844,94.37289429,-222.12469482,-137.94497681,-38.45175552};
    double moments_dg1_dg2[] = {19.27899551,10.84328461,4.37508869,18.64705276,8.82161713,-8.24371624,7.48505402};
    double moments_dg2_dg2[] = {-1013.59936523,-548.21398926,76.33970642,123.20889282,-636.43383789,-195.60177612,-115.77359009};
    double moments_dmu_dg1[] = {-80.75537109,-9.10003662,-1859.73547363,10.84328461,-2.25085449,73.1897583,-149.13095093};
    double moments_dmu_dg2[] = {-265.58776855,-148.86889648,10.84328461,-2013.75817871,-92.80126953,15.95877075,3.05108261};
    double moments_dmu_dmu[] = {-604.41210938,-569.67773438,25.27459717,-55.64135742,-626.43554688,76.24084473,165.08972168};
									      
    cout << "---------- Testing template table ----------" << endl;
    // Read a template table, print out a value
    TemplateTable<BC> tab3(templateIn);
    auto tmpl = tab3.getTemplate(index);
    if (tmpl.id != id) {
      cout << "========>FAILURE: ID read: " << tmpl.id << " Reference " << id << endl;
      failure = true;
    }
    cout << "---ndA: " << endl;
    failure = compare(tmpl.nda, nda, "Read", "Reference") || failure;
    cout << "---jSuppression: " << endl;
    failure = compare(tmpl.jSuppression, jSuppression, "Read", "Reference") || failure;

    auto tab4 = FITS::FitsTable(templateIn,FITS::ReadOnly,1).extract();
    
    cout << "---Derivatives retrieved from file" << endl;
    // Compare tabulated retrieved derivatives to reference values
    failure = compareDerivs2(tab4, moments, "moments", index) || failure;
    failure = compareDerivs2(tab4, moments_dg1, "moments_dg1", index) || failure;
    failure = compareDerivs2(tab4, moments_dg2, "moments_dg2", index) || failure;
    failure = compareDerivs2(tab4, moments_dmu, "moments_dmu", index) || failure;
    failure = compareDerivs2(tab4, moments_dg1_dg1, "moments_dg1_dg1", index) || failure;
    failure = compareDerivs2(tab4, moments_dg1_dg2, "moments_dg1_dg2", index) || failure;
    failure = compareDerivs2(tab4, moments_dg2_dg2, "moments_dg2_dg2", index) || failure;
    failure = compareDerivs2(tab4, moments_dmu_dg1, "moments_dmu_dg1", index) || failure;
    failure = compareDerivs2(tab4, moments_dmu_dg2, "moments_dmu_dg2", index) || failure;
    failure = compareDerivs2(tab4, moments_dmu_dmu, "moments_dmu_dmu", index) || failure;

    cout << "---Derivatives interpreted from TemplateGalaxy" << endl;
    // Get real-valued derivatives, check agreement with inputs.
    failure = compareDerivs(tab4, tmpl, "moments", BC::P, index) || failure;
    failure = compareDerivs(tab4, tmpl, "moments_dg1", BC::DG1, index) || failure;
    failure = compareDerivs(tab4, tmpl, "moments_dg2", BC::DG2, index) || failure;
    failure = compareDerivs(tab4, tmpl, "moments_dmu", BC::DMU, index) || failure;
    failure = compareDerivs(tab4, tmpl, "moments_dg1_dg1", BC::DG1_DG1, index) || failure;
    failure = compareDerivs(tab4, tmpl, "moments_dg1_dg2", BC::DG1_DG2, index) || failure;
    failure = compareDerivs(tab4, tmpl, "moments_dg2_dg2", BC::DG2_DG2, index) || failure;
    failure = compareDerivs(tab4, tmpl, "moments_dmu_dg1", BC::DMU_DG1, index) || failure;
    failure = compareDerivs(tab4, tmpl, "moments_dmu_dg2", BC::DMU_DG2, index) || failure;
    failure = compareDerivs(tab4, tmpl, "moments_dmu_dmu", BC::DMU_DMU, index) || failure;
    
  } catch (std::runtime_error& m) {
    cout << "FAILURE - exception: " << m.what() << endl;
    failure = true;
  }
  if (failure) {
    cout << "FAILURE" << endl;
    exit(1);
  } else {
    exit(0);
  }
}

