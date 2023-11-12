// Repository of string names for columns and keywords in BFD tables
// SYNC THIS TO bfd/keywords.py ******
namespace bfd {
  const map<string, const string> colnames {
    {"tierNumber",  "TIER_NUM"},
    {"covariance", "COVARIANCE"},
    {"pqr", "PQR"},
    {"pqrSel", "PQR_SEL"},
    {"nUnique", "NUNIQUE"},
    {"area", "AREA"},
    {"nodeID","NODE_ID"}};

  const map<string, const string> hdrkeys {
    {"weightN", "WT_N"},
    {"weightSigma", "WT_SIGMA"},
    {"fluxMin", "FLUX_MIN"},
    {"fluxMax", "FLUX_MAX"},
    {"tierNumber", "TIER_NUM"},
    {"sigmaXY", "SIG_XY"},
    {"sigmaFlux", "SIG_FLUX"},
    {"sigmaMax", "SIG_MAX"},
    {"sigmaStep", "SIG_STEP"},
    {"effectiveArea", "EFFAREA"}};
};
