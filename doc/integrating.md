# Notes on the C++ integration software

## Some concepts
Any integration will require a catalog of _target_ galaxies and a catalog of _template_ galaxies to compare them to.
The principle output will be `PQR` values for each target, giving the (log) probability of the measured moments of the target
under the prior that the population of galaxies is described by the template catalog.  Q and R are the first and second derivatives
(vector and matrix, respectively) of the log probability under gravitational lensing.  There are some standard FITS-table formats
for exchanging such catalogs, as defined in `include/MomentTable.h`.  Right now the code specifying columns etc must be manually kept
in sync with the Python code `bfd/momentcalc.py` that writes such FITS tables in order to define the same format.

All moments are measured under a *weight function*.  Our standard weight function (see `KSigmaWeight` class in `include/KGalaxy.h`)
is specified by parameters `(n,sigma)`.  The targets and templates must have been measured under identical weight functions.

Each target or template has measured *moments*.  The moment vector contains the following moments:
* Always present: `MF` (flux), `MR` (quadratic radial), `M1,M2` (quadratic elliptical)
* Present for templates, assumed zero for targets: `MX,MY` (linear/centroid).  Note these are in a separate vector for "odd" moments.
* Optional: `MC` (concentration or 4th radial moment), `MC0` (first color moment, there can be more).

The `Pqr` structure holds 1st and 2nd derivatives w.r.t. lensing of a target's likelihood, and we also will need as input the derivatives of 
each template's moments under lensing.  These derivatives are indexed symbolically by the values of `DG1,DG2,DMU` which represent
derivatives with respect to shear g_1 and g_2, and (optionally) to magnification mu.

In the C++ code, all the values of these indices within the `Moment` and `Pqr` structures are 
specfied in the `BfdConfig` structure (see `include/BfdConfig.h`).
These symbolic indices will have different values depending upon which of the optional moments are in use.  The `BfdConfig` structure
is templated by specifying the values of these parameters (with their assumed types and defaults shown):
* `bool FIX_CENTER=false`: If `true` then the X & Y moments are assumed to be zero, i.e. the object centers were known beforehand.
Obviously this isn't used for real data, just in some testing.
* `bool USE_CONC = false`: Is the `MC` concentration moment in use?
* `bool USE_MAG = false`: Are magnification derivatives included?
* `int N_COLORS=0`: how many color measurements are there?
* `bool USE_FLOAT=true`: are the moment data being kept as `float` or `double`?
Each executable code must specify these decisions and build a `BfdConfig` at the outset, and this structure is then used
as a template argument when declaring most of the BFD classes.  We want to compile these choices into the code so that efficient
small-vector routines can be used.

The target moments have measurement noise, specified by a *covariance matrix*.  The `TargetTable` class allows one either to assume
that all targets have the same covariance matrix (which they will if they all have the same PSF and noise) or if each row of the 
table contains a distinct covariance matrix for its target.

## The KdTree, and matching targets and templates
We do not want to compare _every_ target to _every_ template - this would be painfully slow, and also unnecessary since
(a) most of the template have very little probability of being the truth for a given target, and
(b) we can get an accurate enough Bayesian integration by summing against a random subsample of the templates that are 
feasible matches.

The C++ code selects a subset of the templates to compare to each target.  This is done by placing all of the templates into a kdtree
structure, using as a metric a nominal target covariance matrix.  For each target, we locate all tree nodes whose member templates
are within some `priorSigmaCutoff` of the target's moments (with some slop of `priorSigmaBuffer` similar to the `binslop` parameter 
in Mike Jarvis's `treecorr` code).  In order for this to work efficiently, **all of the targets should have similar covariance matrices,
and the kdtree needs to be built for this nominal covariance.**

There is another reason that the template catalog needs to know about the covariance matrix of the targets. This is because
each template galaxy measured at a grid of (x,y) centroids in order to marginalize over the true center of the target.  The
spacing of this grid is chosen to match the expected noise level in the `MX,MY` moments of the targets.  Furthermore we
construct rotated copies of every template, and the spacing of these rotations is also dependent on the expected target noise.
If we've sampled the templates assuming a noise level that is mis-matched to the targets' noise, then this gridded approximation
to a continuous integral over center and rotation will become inaccurate and/or inefficient.

Finally there is the issue of flux thresholds: the selection function for BFD shear estimates is posed as a minumum `MF`. When
building templates, we discard objects which would never exceed this minimum flux.  So we can't build a template catalog for
one noise level and then apply it to a lower-noise target image where we might want to keep lower-flux targets.

Because of this need for the templates and targets to have roughly the same moment covariance, the nominal covariance values
are stored into the headers of both the target and template catalogs.  There is a check built into
the baseline executable `src/tableIntegrate.cpp` which requires that the input catalogs match within about 5%.

## The nominal executable
The `src/tableIntegrate.cpp` code is the baseline executable for the Bayesian integration.  It has quite a few parameters, which
can either be specified on the command line, or in an ASCII configuration file that is provided as an argument in the command line
or at standard input. Every time you run the `bin/tableIntegrate`, you get the parameter list at stderr:
```
#Wed Jan 20 21:41:27 2021:  bin/tableIntegrate
targetFile  =                     ;Input target moment table filename
templateFile=                     ;Input template moment table filename
outFile     =                     ;(optional) output template moment table filename
doSelection = T                   ;Assign deselection probability to lost and unselected targets?
noiseFactor = 1.                  ;Noise boost factors for kernel smoothing
selectSN    = 0,0                 ;S/N bin divisions
PRIOR:      =                     ;Characteristics of the sampled prior
priorSigmaMax= 0                   ;Maximum sigma range when sampling for prior
priorSigmaStep= 0                   ;Step size when sampling for prior
priorSigmaBuffer= 1                   ;Buffer width of KdTreePrior (in sigma)
nSample     = 30000               ;Number of templates sampled per target (0=all)
maxLeaf     = 0                   ;Maximum number of templates in leaf nodes (0 to default)
sampleWeights= T                   ;Sample templates by weight (T) or number (F)?
minUniqueTemplates= 1                   ;Min number of templates used for valid target
COMPUTING:  =                     ;Configure the computation, usually not needed
nThreads    = -1                  ;Number of threads to use (-1=all)
chunk       = 100                 ;Batch size dispatched to each thread
seed        = 0                   ;Random number seed (0 uses time of day)
```
This is actually an acceptable format for the parameter files (except for the lines with subheadings `PRIOR, COMPUTING`).

And here is the help text:
```
  "tableIntegrate: Integrate targets with moments given in FITS table against templates \n"
  "   given in a different FITS table using KdTree integration.\n"
  "   All targets are assumed to have the same moment covariance matrix.\n"
  "   The *selectSN* parameter is a comma-separated list of values defining a series of\n"
  "     intervals in flux S/N that into which targets will be divided.  Zero means no bound,\n"
  "     and '0,0' is the default, meaning no selection is done.\n"
  "   The *noiseFactor* parameter is a comma-separated list of noise inflation factors\n"
  "     that are applied to each interval, so there should be 1 fewer of these than\n"
  "     entries in the selectSN list.  Default is 1, no added noise.\n"
  "   If *doSelection*=true, a PQR factor is calculated for the probability of non-selection\n"
  "     within the outer selectSN bounds and installed in the output table for these.\n"
  "   The stdout of the program is the total log posterior for lensing assuming postage-stamp\n"
  "     statistics, and including deselection effects if doSelection=true.  The NLOST value\n"
  "     in the targetfile header also is taken to count additional unselected stamps.\n"
  "   PQR values calculated for each target are added to any PQR's already in the targetfile.\n"
  "   If *outfile* parameter is given, the updated target table is written to it.\n"
  "   The *priorSigmaStep, priorSigmaMax* values are taken from templatefile unless overridden by\n"
  "     specification of positive values in the parameters.\n"
  "Usage: tableIntegrate [parameter file] [-parameter [=] value...]\n"
  "       Parameter files read in order of command line, then command-line parameters,\n"
  "       later values override earlier.\n"
  "stdin: none\n"
  "     Run with -help to get parameter list\n"
  "stdout: The summed P,Q,R of the targets and the estimated shear & uncertainty.";
```
Here is some more detail:
* `targetFile, templateFile` - paths to the two primary input files.  Required.
* `outfile`: If you give a file here, it will be the input `targetfile` with additional 
columns created (if needed) that hold the `PQR` values calculated for each galaxy.
Whether or not you enter something here, the stdout of the program is an ASCII set of 
values for the mean lensing shear of the whole target catalog, and its uncertainty.

_to be continued..._
