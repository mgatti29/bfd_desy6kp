#ifndef BFDCONFIG_H
#define BFDCONFIG_H

#include "Std.h"
// The LinearAlgebra.h will choose TMV vs Eigen and define generic types
// that we will use below to make the BFD typedefs.
#include "LinearAlgebra.h"

/************
The BfdConfig contains constants that define the type of BFD calculation being done, and 
defines indices for each of the moments and derivatives in use.  Also defines the TMV
data types that will be used for moments, covariances, etc.
The BfdConfig class has no members so it costs nothing to make an instance.
We only use static, const members so the compiler can know all values.
It will be used as a template argument for all of the BFD classes, so that these choices
are burned in at compile time.

Thus any executable is likely to begin with some code like

#include BfdConfig.h
const bool USE_FLOAT = true;
const bool CENTROID_FIXED = false;
const int NCOLORS = 1;
....
typedef bfd::BfdConfig<USE_CONC, NCOLORS...> CI;

and then can make use of constants and classes CI::MC, CI::MVector, ...
and typedef the BFD classes like

typedef TargetGalaxy_t<CI> TargetGalaxy;

***********/
namespace bfd {
  // First is a templated class that is either float or double
  template <bool UseFloat>
  struct FPType;
  template<>
  struct FPType<true> {
    typedef float FP;
  };
  template<>
  struct FPType<false> {
    typedef double FP;
  };

  template<bool FIX_CENTER=false,
	   bool USE_CONC = false,
	   bool USE_MAG = false,
	   int N_COLORS=0,
	   bool USE_FLOAT=true>
  struct BfdConfig {
    // First make template conditions available as constants
    static const bool FixCenter = FIX_CENTER; // True if XY moments always 0
    static const bool UseConc = USE_CONC;         // True if using concentration moment
    static const bool UseMag = USE_MAG;           // True if magnification being estimated
    static const int Colors = N_COLORS;           // Number of additional flux moments
    static const bool HasColor = N_COLORS>0;      // Any additional flux moments at all?

    // Define indices for individual real-valued moments.
    // Non-existent moments have index<0
    static const int MF = 0;                      // Flux moment
    static const int MR = MF+1;                   // Radius/size moment |k^2|
    static const int M1 = MR+1;                   // 1st ellipticity moment
    static const int M2 = M1+1;                   // 2nd ellipticity moment
    static const int MC = UseConc ? M2+1 : -1;    // Concentration moment |k^4|
    static const int MC0 = UseConc ? MC+1 : M2+1; // Index of first color
    static const int MSIZE = MC0 + Colors;        // Total size of (non-centroid) moment vector

    static const int MX = FixCenter ? -1 : 0;     // Index of X moment in XY vector
    static const int MY = FixCenter ? -1 : 1;     // Index of Y moment in XY vector
    static const int XYSIZE = FixCenter ? 1 : 2;  // Size of XY moment vector - must be >0 for tmv::SmallVector

    static const int MXYSIZE = FixCenter ? MSIZE : MSIZE+XYSIZE; // all moments

    // Indices for Taylor expansions w.r.t. lensing
    static const int P = 0;                       // Index of 0th derivative, i.e. value
    static const int Q = 1;                       // Index where 1st derivs start
    static const int DG1 = Q;                     // Deriv wrt G1
    static const int DG2 = DG1+1;                 // Deriv wrt G2
    static const int DMU = UseMag ? DG2+1 : -1;   // Deriv wrt magnification
    static const int ND = UseMag ? 3 : 2;         // Number of lensing 1st derivs
    static const int R = Q + ND;                  // Index where 2nd derivs begin
    static const int DG1_DG1 = R;                  // 2nd derivative
    static const int DG1_DG2 = DG1_DG1+1;           // 2nd derivative
    static const int DG2_DG2 = DG1_DG2+1;           // 2nd derivative
    static const int DMU_DMU = UseMag ? DG2_DG2+1 : -1; 
    static const int DMU_DG1 = UseMag ? DMU_DMU+1 : -1; 
    static const int DMU_DG2 = UseMag ? DMU_DG1+1 : -1;
    static const int DSIZE = UseMag ? DMU_DG2+1 : DG2_DG2+1; // Total dimension of Taylor expansion

    // Indices for a lensing vector components (QVector or RMatrix):
    static const int G1 = 0;                    // Deriv wrt G1
    static const int G2 = G1+1;                 // Deriv wrt G2
    static const int MU = UseMag ? G2+1 : -1;   // Deriv wrt magnification
    
    // Define data types
    typedef typename FPType<USE_FLOAT>::FP FP;
    typedef linalg::SVector<FP, MSIZE> MVector;    // Moment vector (no centroid)
    typedef linalg::SVector<FP, XYSIZE> XYVector;  // Centroid moment vector
    typedef linalg::SVector<FP, MXYSIZE> MXYVector;  // all-moments vector
    typedef linalg::SVector<FP, DSIZE> PqrVector;  // Taylor expansion vector
    typedef linalg::SMatrix<FP, MSIZE, MSIZE> MMatrix; // Moment covariance or transform
    typedef linalg::SMatrix<FP, XYSIZE, XYSIZE> XYMatrix;    //  XY moment covariance or transform
    typedef linalg::SMatrix<FP, MXYSIZE, MXYSIZE> MXYMatrix; // Full moment xform
    typedef linalg::SVector<FP, ND> QVector;       // 1st derivs wrt lensing
    typedef linalg::SMatrix<FP, ND, ND> RMatrix;   // 2nd derivs wrt lensing
    typedef linalg::SMatrix<FP, MSIZE, DSIZE> MDMatrix; // Moment all derivative matrix
    typedef linalg::SMatrix<FP, XYSIZE, DSIZE> XYDMatrix; // XY derivative matrix
    // Dynamic-sized vector & matrix of appropriate types:
    typedef linalg::Vector<FP> Vector;
    typedef linalg::Matrix<FP> Matrix;
  };

} // end namespace bfd

#endif   // BFDCONFIG_H
