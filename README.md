# bfd
Bayesian Fourier Domain weak lensing measurement

Contains Python codes for measuring the PSF-corrected Fourier domain moments of galaxies from pixelized images, and C++ codes
for derivation of gravitational lensing constraints from comparison of these moments to those measured on a set of high-S/N "template"
galaxies.  Also in C++ code one can acquire analytic moments for Gaussian galaxies, and there is a set of alternative pixel-measurement
codes that is currently disabled.

See [Bernstein & Armstrong (2014)](http://adsabs.harvard.edu/abs/2014MNRAS.438.1880B) 
and [Bernstein _et al._ (2016)](http://adsabs.harvard.edu/abs/2016MNRAS.459.4467B) 
for the derivations and explanations of the method.  LATeX source for the latter is in [the doc/ directory](doc/).

The Python and C++ codes do not call each other and can be built/installed independently.  They are in the same repo because
the interchange of information between them requires shared conventions about file formats and indices, etc.

## Prerequistes

*Python:* Nothing special to use the module: numpy, scipy, astropy. Specialized external modules needed for the codes that
access particular projects' data formats, e.g. [BFDMEDS](https://github.com/danielgruen/bfdmeds).

*C++:* 

* [FFTW](http://www.fftw.org)
* [TMV](https://github.com/rmjarvis/tmv) -OR-
* [Eigen](http://eigen.tuxfamily.org)
* [cfitsio](http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html)
* [yaml-cpp](https://github.com/jbeder/yaml-cpp)
* [gbutils](../gbtools)
* [gbfits](../gbfits)
* [sbprofile](../sbprofile) (once the C++ pixel-oriented routines are re-enabled)

## Installation

See the [INSTALL file](INSTALL)

## Usage, Coding:

No documentation yet, sorry.
