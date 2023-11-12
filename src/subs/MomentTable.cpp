#include "MomentTable.h"
#include "FitsTable.h"
#include "FitsImage.h"

#include "keywords.h"
using namespace bfd;
/*
using namespace bfd;
// Target table column names
const string tierNumberColumn = "TIER_NUM";
const string covarianceColumn = "COVARIANCE";
const string pqrColumn = "PQR";
const string pqrSelColumn = "PQR_SEL";
const string nUniqueColumn = "NUNIQUE";
const string areaColumn = "AREA";

// Header keywords
const string weightNKey = "WT_N";
const string weightSigmaKey = "WT_SIGMA";
const string fluxMinKey = "FLUX_MIN";
const string fluxMaxKey = "FLUX_MAX";
const string tierNumberKey = "TIER_NUM";
const string sigmaXYKey = "SIG_XY";
const string sigmaFluxKey = "SIG_FLUX";
const string sigmaMaxKey = "SIG_MAX";
const string sigmaStepKey = "SIG_STEP";
const string effectiveAreaKey = "EFFAREA";
*/

// Declare the static map giving Python indices for moments
template <class CONFIG>
vector<int>
TargetTable<CONFIG>::iMap;

template <class CONFIG>
void
TargetTable<CONFIG>::buildIMap() {
  // Create the static Python index map, if not done yet.
  if (iMap.empty()) {
    iMap.resize(BC::MSIZE);
    if (BC::MF>=0) iMap[BC::MF] = M0;
    if (BC::MR>=0) iMap[BC::MR] = MR;
    if (BC::M1>=0) iMap[BC::M1] = M1;
    if (BC::M2>=0) iMap[BC::M2] = M2;
    if (BC::MC>=0) iMap[BC::MC] = MC;
  }
}

// Helper functions for TemplateSpecs:
double get(const img::Header& h, string kw) {
  // Get a double-valued header entry or -1 if it's absent
  if (h.find(kw)) {
    double d;
    h.getValue(kw,d);
    return d;
  } else {
    return -1.;
  }
}
void put(img::Header& h, string kw, double val) {
  // If value is negative and no previous kw is present,
  // we can just leave this out of the header as undefined.
  if (val>0 || h.find(kw)) {
    h.replace(kw, val);
  }
}

TemplateSpecs::TemplateSpecs(const img::Header& h) {
  // Construct a TemplateSpecs instance from header entries
  double d;
  h.getValue(hdrkeys.at("weightN"),d);
  weightN = round(d);
  h.getValue(hdrkeys.at("weightSigma"),weightSigma);
  fluxMin = get(h, hdrkeys.at("fluxMin"));
  fluxMax = get(h, hdrkeys.at("fluxMax"));
  sigmaXY = get(h, hdrkeys.at("sigmaXY"));
  sigmaFlux = get(h, hdrkeys.at("sigmaFlux"));
  sigmaMax = get(h, hdrkeys.at("sigmaMax"));
  sigmaStep = get(h, hdrkeys.at("sigmaStep"));
}

void
TemplateSpecs::addToHeader(img::Header& h) const {
  // Write specs to a FITS header (if they are defined)
  h.replace(hdrkeys.at("weightN"), weightN);
  h.replace(hdrkeys.at("weightSigma"), weightSigma);
  put(h, hdrkeys.at("fluxMin"), fluxMin);
  put(h, hdrkeys.at("fluxMax"), fluxMax);
  put(h, hdrkeys.at("sigmaXY"), sigmaXY);
  put(h, hdrkeys.at("sigmaFlux"), sigmaFlux);
  put(h, hdrkeys.at("sigmaMax"), sigmaMax);
  put(h, hdrkeys.at("sigmaStep"), sigmaStep);
}

template <class CONFIG>
MomentCovariance<CONFIG>
TargetTable<CONFIG>::unpackCov(vector<float> cov) {
  // Build covariance from the saved triangle of even:

  // Make sure the map from python indices exists
  buildIMap();

  MomentCovariance<BC> out;
  // Order of coefficient must match convention of momenttable.py!!!
  // First build a symmetric matrix in the stored (python) order
  linalg::Matrix<FP> even_in(NM,NM);
  int k=0;
  for (int i=0; i<NM; i++) {
    for (int j=i; j<NM; j++, k++) {
      even_in(i,j) = even_in(j,i) = cov[k];
    }
  }
  // Then copy the moments we're using into C++-order matrix
  for (int i=0; i<BC::MSIZE; i++)
    for (int j=0; j<BC::MSIZE; j++)
      out.m(i,j) = even_in(iMap[i],iMap[j]);

  // And the XY covariance, if in use, comes from some elements of even
  if (!BC::FixCenter) {
    out.xy(BC::MX,BC::MX) = 0.5*(out.m(BC::MF,BC::MR)+out.m(BC::MF,BC::M1));
    out.xy(BC::MX,BC::MY) = 0.5*out.m(BC::MF,BC::M2);
    out.xy(BC::MY,BC::MX) = 0.5*out.m(BC::MF,BC::M2);
    out.xy(BC::MY,BC::MY) = 0.5*(out.m(BC::MF,BC::MR)-out.m(BC::MF,BC::M1));
  }
  return out;
}

template <class CONFIG>
MomentCovariance<CONFIG>
TargetTable<CONFIG>::getCov(long index) const {
  // Build one from the saved triangle:
  vector<float> vf;
  tab.readCell(vf, colnames.at("covariance"), index);
  return unpackCov(vf);
}

template <class CONFIG>
TargetTable<CONFIG>::TargetTable(string fitsname, bool update):
  fitsTabPtr(nullptr)
{
  // Create the static Python index map, if not done yet.
  buildIMap();

  // Read the table, assumed in the first extension
  if (update) {
    // Keep the FITS connection open
    fitsTabPtr = new FITS::FitsTable(fitsname, FITS::ReadWrite);
    tab = fitsTabPtr->use();
  } else {
    // extract table and close FITS
    FITS::FitsTable tmp(fitsname, FITS::ReadOnly);
    tab = tmp.extract();
  }

  // If there is no cov_even column in the table, we assume
  // all will have their tier's covariance.
  try {
    long csize = tab.repeat(colnames.at("covariance"));
    sharedCov = false;
    // Check proper size of the columns
    if (csize != NM*(NM+1)/2)
      throw std::runtime_error("MomentsTable cov_even column is not of expected width");
  } catch (img::FTableNonExistentColumn &m) {
    sharedCov = true;
  }

  // Extract some properties from header
  {
    TemplateSpecs ts(*tab.header());
    wtN = ts.weightN;
    wtSigma = ts.weightSigma;
  }

  // See if our table has tier indices or not
  hasTierColumn = tab.hasColumn(colnames.at("tierNumber"));
}


template <class CONFIG>
TargetTable<CONFIG>::~TargetTable() {
  // Get rid of the table
  tab = img::FTable();
  if (fitsTabPtr)
    // Delete the saved FITS references to trigger write to disk
    delete fitsTabPtr;
}

    
template <class CONFIG>
TargetGalaxy<CONFIG>
TargetTable<CONFIG>::getTarget(long index) const {
  typename BC::MVector m;
  vector<float> vf;
  tab.readCell(vf, "moments", index);
  for (int j=0; j<BC::MSIZE; j++)
    m[j] = vf[iMap[j]];
  linalg::DVector2 xy;
  try {
    vector<double> vd;
    tab.readCell(vd,"xy",index);
    for (int j=0; j<2; j++)
      xy[j] = vd[j];
  } catch (img::FTableError& e) {
    // Might be single precision
    tab.readCell(vf,"xy",index);
    for (int j=0; j<2; j++)
      xy[j] = vf[j];
  }
  LONGLONG id;
  tab.readCell(id, "id", index);
  return TargetGalaxy<BC>(Moment<BC>(m), getCov(index), xy, id);
}


template <class CONFIG>
void
TargetTable<CONFIG>::setPqr(const Pqr<BC>& pqr, long index) {
  checkPqrColumn();
  vector<float> vf(BC::DSIZE);
  for (int i=0; i<BC::DSIZE; i++)
    vf[i] = pqr[i];
  tab.writeCell(vf, colnames.at("pqr"), index);
}

template <class CONFIG>
void
TargetTable<CONFIG>::addPqr(const Pqr<BC>& pqr, long index) {
  vector<float> vf(BC::DSIZE);
  tab.readCell(vf, colnames.at("pqr"), index);
  for (int i=0; i<BC::DSIZE; i++)
    vf[i] += pqr[i];
  tab.writeCell(vf, colnames.at("pqr"), index);
}

template <class CONFIG>
Pqr<CONFIG>
TargetTable<CONFIG>::getPqr(long index) const {
  vector<float> vf(BC::DSIZE);
  tab.readCell(vf, colnames.at("pqr"), index);
  Pqr<BC> out;
  for (int i=0; i<BC::DSIZE; i++)
    out[i] = vf[i];
  return out;
}

template <class CONFIG>
void
TargetTable<CONFIG>::checkPqrColumn() {
  bool haveIt = false;
  for (auto colname : tab.listColumns()) {
    if (colname==colnames.at("pqr")) {
      if (tab.repeat(colname)!=BC::DSIZE)
	throw std::runtime_error("Invalid width of PQR column in FITS table");
      haveIt = true;
      break;
    }
  }
  if (!haveIt) {
    vector<float> vzero(BC::DSIZE, 0.);
    vector<vector<float>> vpqr(tab.nrows(), vzero);
    tab.addColumn(vpqr, colnames.at("pqr"), BC::DSIZE);
  }
}

template <class CONFIG>
void
TargetTable<CONFIG>::checkNUniqueColumn() {
  if (tab.hasColumn(colnames.at("nUnique"))) return;
  vector<int> v(tab.nrows(), 0);
  tab.addColumn(v, colnames.at("nUnique"));
}

template <class CONFIG>
int
TargetTable<CONFIG>::getNUnique(long index) const {
  int nUnique;
  tab.readCell(nUnique, colnames.at("nUnique"), index);
  return nUnique;
}

template <class CONFIG>
void
TargetTable<CONFIG>::setNUnique(int nUnique, long index) {
  checkNUniqueColumn();
  tab.writeCell(nUnique, colnames.at("nUnique"), index);
}

template <class CONFIG>
void
TargetTable<CONFIG>::checkAreaColumn() {
  if (tab.hasColumn(colnames.at("area"))) return;
  vector<float> v(tab.nrows(), 0);
  tab.addColumn(v, colnames.at("area"));
}

template <class CONFIG>
double
TargetTable<CONFIG>::getArea(long index) const {
  float area;
  tab.readCell(area, colnames.at("area"), index);
  return area;
}

template <class CONFIG>
void
TargetTable<CONFIG>::setArea(double area, long index) {
  checkAreaColumn();
  tab.writeCell((float) area, colnames.at("area"), index);
}

template <class CONFIG>
int
TargetTable<CONFIG>::getTier(long index) const {
  int tier = 0;
  /**/LONGLONG long_tier = 0;
  if (hasTierColumn)
    tab.readCell(long_tier,colnames.at("tierNumber"),index);
  return long_tier;
}

template <class CONFIG>
void
TargetTable<CONFIG>::addHistory(string h) {
  tab.header()->addHistory(h);
}

template <class CONFIG>
void
TargetTable<CONFIG>::save(string fitsname) const {
  // Write the table (will create null primary extension)
  FITS::FitsTable ft(fitsname, FITS::ReadWrite+FITS::Create+FITS::OverwriteFile, 1);
  ft.copy(tab);
}

//****************** NoiseTier stuff ***********************

template <class CONFIG>
int
NoiseTier<CONFIG>::getID() const {
  int id;
  tab.header()->getValue(hdrkeys.at("tierNumber"),id);
  return id;
}

template <class CONFIG>
void
NoiseTier<CONFIG>::setPqr(const Pqr<BC>& pqr, int index) {
  checkPqrColumn();
  vector<float> vf(BC::DSIZE);
  for (int i=0; i<BC::DSIZE; i++)
    vf[i] = pqr[i];
  tab.writeCell(vf, colnames.at("pqrSel"), index);
}

template <class CONFIG>
Pqr<CONFIG>
NoiseTier<CONFIG>::getPqr(int index) const {
  vector<float> vf(BC::DSIZE);
  tab.readCell(vf, colnames.at("pqrSel"), index);
  Pqr<BC> out;
  for (int i=0; i<BC::DSIZE; i++)
    out[i] = vf[i];
  return out;
}

template <class CONFIG>
void
NoiseTier<CONFIG>::checkPqrColumn() {
  bool haveIt = false;
  for (auto colname : tab.listColumns()) {
    if (colname==colnames.at("pqrSel")) {
      if (tab.repeat(colname)!=BC::DSIZE)
	throw std::runtime_error("Invalid width of PQR column in NoiseTier table");
      haveIt = true;
      break;
    }
  }
  if (!haveIt) {
    vector<float> vzero(BC::DSIZE, 0.);
    vector<vector<float>> vpqr(tab.nrows(), vzero);
    tab.addColumn(vpqr, colnames.at("pqrSel"), BC::DSIZE);
  }
}

template <class CONFIG>
MomentCovariance<CONFIG>
NoiseTier<CONFIG>::getCov(int index) const {
  // Build one from the saved triangle:
  vector<float> vf;
  tab.readCell(vf, colnames.at("covariance"), index);
  return TargetTable<BC>::unpackCov(vf);
}


template <class CONFIG>
void
NoiseTier<CONFIG>::addHistory(string h) {
  tab.header()->addHistory(h);
}
    
//************** NoiseTierCollection ****************

template<class CONFIG>
NoiseTierCollection<CONFIG>::NoiseTierCollection(string filename,
						 bool update) {
  // Step through the FITS extensions looking for NoiseTier tables
  if (update) {
    // Open everything as READWRITE
    FITS::FitsFile ff(filename, FITS::ReadWrite);
    for (int i=0; i<ff.HDUCount(); i++) {
      if (ff.getHDUType(i)==FITS::HDUBinTable) {
	auto ft = new FITS::FitsTable(filename, FITS::ReadWrite, i);
	if (ft->getName().substr(0,4)=="TIER") {
	  // This is a noise tier we want to use and update
	  auto nt = new NoiseTier<BC>(ft->use());
	  tiers[nt->getID()] = nt;
	  // Save the fits handle to delete later and force update
	  savedFITS.push_back(ft);
	  /**/cerr << "Added tier " << nt->getID() << endl;
	} else {
	  // Not a noise tier
	  delete ft;
	}
      }
    }
  } else {
    // Not updating outputs; open everything read-only
    FITS::FitsFile ff(filename, FITS::ReadOnly);
    for (int i=0; i<ff.HDUCount(); i++) {
      if (ff.getHDUType(i)==FITS::HDUBinTable) {
	FITS::FitsTable ft(filename, FITS::ReadOnly, i);
	if (ft.getName().substr(0,4)=="TIER") {
	  // This is a noise tier we want to use and update
	  auto nt = new NoiseTier<BC>(ft.extract());
	  tiers[nt->getID()] = nt;
	}
      }
    }
  }
}

template<class CONFIG>
NoiseTierCollection<CONFIG>::~NoiseTierCollection() {
  // Delete all NoiseTiers
  for (auto& pr : tiers) {
    delete pr.second;
  }
  // And any open FITSTable pointers
  for (auto ff : savedFITS) {
    delete ff;
  }
}

// *************************** TemplateTable *******************

template <class CONFIG>
TemplateTable<CONFIG>::TemplateTable(string fitsname) {
  // Read data from table extension
  FITS::FitsTable ft(fitsname, FITS::ReadOnly, 1);
  tab = ft.extract();

  if (!tab.header()->getValue(hdrkeys.at("effectiveArea"),effArea)) {
    // Absent keyword means templates already normed
    effArea = 1.;
  }
}

template <class CONFIG>
int
TemplateTable<CONFIG>::getTier() const {
  int i;
  tab.header()->getValue(hdrkeys.at("tierNumber"),i);
  return i;
}
  

template <class CONFIG>
void
TemplateTable<CONFIG>::readM(linalg::Vector<FP>& out,
			     string colName,
			     long irow) const {
  vector<float> vf(NM);
  tab.readCell(vf, colName, irow);
  for (int i=0; i<NM; i++)
    out[i] = vf[i];
}

template <class CONFIG>
void
TemplateTable<CONFIG>::saveM(linalg::Vector<std::complex<FP>>& z,
			     typename TG::DerivMatrix& mderiv,
			     typename TG::DerivMatrix& xyderiv,
			     int irow) const {
  const std::complex<FP> I(0,1);
  
  mderiv(TG::MF, irow) = z[M0];
  mderiv(TG::MR, irow) = z[MR];
  mderiv(TG::ME, irow) = z[M1] + I*z[M2];
  if (BC::UseConc)
    mderiv(TG::MC, irow) = z[MC];
  if (!BC::FixCenter)
    xyderiv(TG::MX,irow) = z[MX]+I*z[MY];
}
  
template <class CONFIG>
TemplateGalaxy<CONFIG>
TemplateTable<CONFIG>::getTemplate(long index) const {
  typename TG::DerivMatrix mderiv(TG::MSIZE,TG::DSIZE,0.);
  typename TG::DerivMatrix xyderiv(TG::XYSIZE,TG::DSIZE,0.);

  // transform real derivs back into complex
  // Some scratch space
  vector<float> vf;
  linalg::Vector<std::complex<FP> > z(NM);
  linalg::Vector<FP> r1(NM), r2(NM), r3(NM);
  const std::complex<FP> I(0,1);

  readM(r1, "moments", index);
  z = r1;
  saveM(z, mderiv, xyderiv, TG::D0);

  readM(r1, "moments_dg1", index);
  readM(r2, "moments_dg2", index);
  z = FP(0.5) * (r1 - I * r2);
  saveM(z, mderiv, xyderiv, TG::DV);
  z = FP(0.5) * (r1 + I * r2);
  saveM(z, mderiv, xyderiv, TG::DVb);
	
  readM(r1, "moments_dg1_dg1", index);
  readM(r2, "moments_dg1_dg2", index);
  readM(r3, "moments_dg2_dg2", index);
  z = FP(0.25) * (r1-r3) - FP(0.5)*I*r2;
  saveM(z, mderiv, xyderiv, TG::DV_DV);
  z = FP(0.25) * (r1-r3) + FP(0.5)*I*r2;
  saveM(z, mderiv, xyderiv, TG::DVb_DVb);
  z = FP(0.25) * (r1+r3);
  saveM(z, mderiv, xyderiv, TG::DV_DVb);
  
  if (BC::UseMag) {
    readM(r1, "moments_dmu", index);
    z = r1;
    saveM(z, mderiv, xyderiv, TG::DU);
    
    readM(r1, "moments_dmu_dg1", index);
    readM(r2, "moments_dmu_dg2", index);
    z = FP(0.5) * (r1 - I * r2);
    saveM(z, mderiv, xyderiv, TG::DU_DV);
    z = FP(0.5) * (r1 + I * r2);
    saveM(z, mderiv, xyderiv, TG::DU_DVb);

    readM(r1, "moments_dmu_dmu", index);
    z = r1;
    saveM(z, mderiv, xyderiv, TG::DU_DU);
  }
  LONGLONG id;
  tab.readCell(id, "id",index);
  float weight;
  tab.readCell(weight, "weight",index);
  float jSuppress;
  tab.readCell(jSuppress, "jSuppress",index);

  return TemplateGalaxy<BC>(mderiv, xyderiv,
			    weight,
			    id,
			    jSuppress);
}

#define INSTANTIATE(...)					\
  template class bfd::TargetTable<BfdConfig<__VA_ARGS__> >;	\
  template class bfd::TemplateTable<BfdConfig<__VA_ARGS__> >;   \
  template class bfd::NoiseTier<BfdConfig<__VA_ARGS__> >;      \
  template class bfd::NoiseTierCollection<BfdConfig<__VA_ARGS__> >;

#include "InstantiateMomentCases.h"
