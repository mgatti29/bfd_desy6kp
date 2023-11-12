// Functions useful for unit tests.
// Much of which probably could be done with something like Boost test.

#ifndef TEST_H
#define TEST_H

#include <iostream>
#include <iomanip>
#include <cmath>
#include "LinearAlgebra.h"
#include "BfdConfig.h"
#include "Moment.h"

using namespace linalg;

namespace test {
  // All functions return true if there is a failure to agree at specified precision
  const double TOLERANCE = 1e-4;
  const double FTOL = 1e-3;

  // Scalar comparison
  bool
    compare(double d1, double d2,
	    string label1="Analytic",string label2="Numeric") {
    bool failure = false;
    double diff = d1 - d2;

    if (abs(diff) > max(TOLERANCE, FTOL*abs(d1))) {
      failure = true;
    }
    if (failure) cout << "========>FAILURE:" << endl;
    int w = std::max(label1.size(), label2.size()) + 2;
    cout << left << setw(w) << label1 + ":" << fixed << setprecision(6) << d1 << endl;
    cout << left << setw(w) << label2 + ":" << fixed << setprecision(6) << d2 << endl;
    return failure;
  }
  // Vectors - or anything class with a size() and indices
  // and values castable to double
  template <class T>
  bool
  compare(const T& d1, const T& d2,
	  string label1="Analytic",string label2="Numeric") {
    int N = d1.size();
    bool failure = false;
    for (int i=0; i<N; i++) {
      double diff = d1[i] - d2[i];
      if (abs(diff) > max(TOLERANCE, FTOL*abs(d1[i]))) {
	failure = true;
      }
    }
    int w = std::max(label1.size(), label2.size()) + 2;
    if (failure) cout << "========>FAILURE:" << endl;
    cout << left << setw(w) << label1 + ":";
    for (int i=0; i<N; i++) cout << fixed << setprecision(6) << d1[i] << " ";
    cout << endl;
    cout << left << setw(w) << label2 + ":";
    for (int i=0; i<N; i++) cout << fixed << setprecision(6) << d2[i] << " ";
    cout << endl;
    return failure;
  }
  // bfd Moment vectors (even and odd moments)
  template <class CONFIG>
    extern bool
    compare(const bfd::Moment<CONFIG>& m1, const bfd::Moment<CONFIG>& m2,
	    string label1="Analytic",string label2="Numeric") {
    typedef CONFIG BC;
    bool failure = false;
    for (int i=0; i<BC::MSIZE; i++) {
      double diff = m1.m[i] - m2.m[i];
      if (abs(diff) > max(TOLERANCE, FTOL*abs(m1.m[i]))) {
	failure = true;
      }
    }
    if (!BC::FixCenter) {
      for (int i=0; i<BC::XYSIZE; i++) {
	double diff = m1.xy[i] - m2.xy[i];
	if (abs(diff) > max(TOLERANCE, FTOL*abs(m1.xy[i]))) {
	  failure = true;
	}
      }
    }
    int w = std::max(label1.size(), label2.size()) + 2;
    if (failure) cout << "========>FAILURE:" << endl;
    cout << left << setw(w) << label1 + ":";
    for (int i=0; i<BC::MSIZE; i++) cout << fixed << setprecision(6) << m1.m[i] << " ";
    if (!BC::FixCenter) 
      for (int i=0; i<BC::XYSIZE; i++) cout << fixed << setprecision(6) << m1.xy[i] << " ";
    cout << endl;
    cout << left << setw(w) << label2 + ":";
    for (int i=0; i<BC::MSIZE; i++) cout << fixed << setprecision(6) << m2.m[i] << " ";
    if (!BC::FixCenter) 
      for (int i=0; i<BC::XYSIZE; i++) cout << fixed << setprecision(6) << m2.xy[i] << " ";
    cout << endl;
    return failure;
  }

} // namespace test

#endif //TEST_H
