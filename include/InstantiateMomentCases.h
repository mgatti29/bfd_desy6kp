/*
Common code listing all the cases of BfdConfig-urations that
we want to instantiate for the templated classes.
Each .cpp should have a 
#define INSTANTIATE(...)
macro defined before including this file which uses the __VA_ARGS__
facility of the macros to insert all of the arguments into a BfdConfig
template, e.g. 
#define INSTANTIATE(...)	\
  template class Moment<BfdConfig<__VA_ARGS__> >;

Note the order of BfdConfig template arguments from BfdConfig.h is
 template<bool FIX_CENTER=false,
	   bool USE_CONC = false,
	   bool USE_MAG = false,
	   int N_COLORS=0,
	   bool USE_FLOAT=true>
***/
// Instantiate "normal" case
INSTANTIATE(false, false, false)
// Fixed-centroid case
INSTANTIATE(true, false, false)
// A case including magnification
INSTANTIATE(false, false, true)
// ...plus concentration moment
INSTANTIATE(false, true, true)
// ...all moments, double precision, used in tests
INSTANTIATE(false, true, true, 0, false)
// Fixed-centroid case w/magnification, no concentratin, more tests
INSTANTIATE(true, false, true)

