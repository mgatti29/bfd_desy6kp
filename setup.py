from __future__ import print_function
import sys,os,glob,re
import select

#from distutils.core import setup
from setuptools import setup
import distutils

print('Python version = ',sys.version)
py_version = "%d.%d"%sys.version_info[0:2]  # we check things based on the major.minor version.

dependencies = ['numpy', 'future', 'astropy', 'scipy']

#with open('README.md') as file:
#    long_description = file.read()

# Read in the version from bfd/_version.py
# cf. http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
version_file=os.path.join('bfd','_version.py')
verstrline = open(version_file, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    bfd_version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (version_file,))
print('BFD version is %s'%(bfd_version))

# data = glob.glob(os.path.join('data','*'))

dist = setup(
        name="BFD",
        version=bfd_version,
        author="Gary Bernstein",
        author_email="garyb@PHYSICS.UPENN.EDU",
        description="Python module for measuring Bayesian Fourier Domain galaxy moments",
#        long_description=long_description,
        license = "BSD License",
        url="https://github.com/gbernstein/bfd",
        download_url="https://github.com/gbernstein/bfd/releases/tag/v%s.zip"%bfd_version,
        packages=['bfd'],
#        package_data={'bfd' : data },
        install_requires=dependencies,
        scripts=['scripts/createTiers.py','scripts/assignSelection.py']
    )

