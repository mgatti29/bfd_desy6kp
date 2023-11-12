# These site-dependent items should be defined in environment:

# CXX 
# CXXFLAGS
# EIGEN_DIR
# CFITSIO_DIR
# GBUTIL_DIR
# GBFITS_DIR
# optional:
# MKL_DIR, MKL_OPT, MKL_LINK


INCLUDES := 

LIBS := -lm

EXTDIRS := 

# Collect the includes and libraries we need
ifdef FFTW_DIR
INCLUDES += -I $(FFTW_DIR)/include
LIBS += -L $(FFTW_DIR)/lib -lfftw3
else
$(error Require FFTW_DIR in environment)
endif

ifdef CFITSIO_DIR
INCLUDES += -I $(CFITSIO_DIR)/include
LIBS += -L $(CFITSIO_DIR)/lib -lcfitsio
else
$(error Require CFITSIO_DIR in environment)
endif

ifdef GBUTIL_DIR
INCLUDES += -I $(GBUTIL_DIR)/include
EXTDIRS += $(GBUTIL_DIR)
GBUTIL_OBJ = $(GBUTIL_DIR)/obj
else
$(error Require GBUTIL_DIR in environment)
endif

ifdef GBFITS_DIR
INCLUDES += -I $(GBFITS_DIR)
EXTDIRS += $(GBFITS_DIR)
else
$(error Require GBFITS_DIR in environment)
endif

ifdef EIGEN_DIR
INCLUDES += -I $(EIGEN_DIR) -D USE_EIGEN
else
$(error Require EIGEN_DIR in environment)
endif

ifdef MKL_DIR
INCLUDES += ${MKL_OPTS}
LIBS += ${MKL_LINK}
endif

# Object files found in external packages:
EXTOBJS =$(GBUTIL_OBJ)/BinomFact.o $(GBUTIL_OBJ)/StringStuff.o $(GBUTIL_OBJ)/Interpolant.o \
	$(GBUTIL_OBJ)/fft.o $(GBUTIL_OBJ)/Table.o $(GBUTIL_OBJ)/Pset.o \
	$(GBUTIL_OBJ)/Expressions.o $(GBUTIL_OBJ)/Shear.o \
	$(GBFITS_DIR)/FITS.o $(GBFITS_DIR)/Header.o $(GBFITS_DIR)/Hdu.o $(GBFITS_DIR)/FitsTable.o \
	$(GBFITS_DIR)/FTable.o $(GBFITS_DIR)/FTableExpression.o \
	$(GBFITS_DIR)/Image.o $(GBFITS_DIR)/FitsImage.o

ifdef SBPROFILE_DIR
INCLUDES += -I $(GBFITS_DIR)/include
EXTDIRS += $(SBPROFILE_DIR)
EXTOBJS += $(SBPROFILE_DIR)/SBProfile.o $(SBPROFILE_DIR)/SBPixel.o 
# ??? Add executables and subroutines for drawing-oriented stuff
endif

##### 
BINDIR = bin
OBJDIR = obj
SRCDIR = src
SUBDIR = src/subs
INCLUDEDIR = include
TESTDIR = tests
TESTBINDIR = testbin


# INCLUDES can be relative paths, and will not be exported to subdirectory makes.
INCLUDES += -I $(INCLUDEDIR)

# Executable C++ programs
EXECS :=  $(wildcard $(SRCDIR)/*.cpp)
TARGETS := $(EXECS:$(SRCDIR)/%.cpp=$(BINDIR)/%)
OBJS := $(EXECS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
# Executable C++ needing special compilation flags
SPECIAL_TARGETS = $(BINDIR)/gaussTest4 $(BINDIR)/gaussTest4m $(BINDIR)/gaussTest6 $(BINDIR)/gaussTest6m
# Python executables
PYEXECS :=  $(wildcard $(SRCDIR)/*.py)
PYTARGETS :=  $(PYEXECS:$(SRCDIR)/%.py=$(BINDIR)/%.py)
# C++ subroutines
SUBS :=  $(wildcard $(SUBDIR)/*.cpp)
SUBOBJS := $(SUBS:$(SUBDIR)/%.cpp=$(OBJDIR)/%.o)

CP = /bin/cp -p
RM = /bin/rm -f

#######################
# Rules - ?? dependencies on INCLUDES ??
#######################

all: cpp python

cpp: exts $(TARGETS) $(SPECIAL_TARGETS)

python: $(PYTARGETS)
	python ./setup.py install

# Compilation
$(OBJS):  $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(SUBOBJS): $(OBJDIR)/%.o : $(SUBDIR)/%.cpp 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/gaussTest6.o: $(SRCDIR)/gaussTest.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -DSHIFT -o $@

$(OBJDIR)/gaussTest4.o: $(SRCDIR)/gaussTest.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR)/gaussTest6m.o: $(SRCDIR)/gaussTest.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -DSHIFT -DMAGNIFY -o $@

$(OBJDIR)/gaussTest4m.o: $(SRCDIR)/gaussTest.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -DMAGNIFY -o $@

# Linking
$(TARGETS): $(BINDIR)/% : $(OBJDIR)/%.o $(SUBOBJS) $(EXTOBJS)
	$(CXX) $(CXXFLAGS) $^  $(LIBS) -o $@

$(SPECIAL_TARGETS): $(BINDIR)/% : $(OBJDIR)/%.o $(SUBOBJS) $(EXTOBJS)
	echo $(SUBOBJS)
	$(CXX) $(CXXFLAGS) $^  $(LIBS) -o $@

# Python executables - copy into bin directory
$(PYTARGETS): $(BINDIR)/% : $(SRCDIR)/%
	$(CP) $^ $@

######### Test programs

TESTSRC := $(wildcard $(TESTDIR)/*.cpp)
TESTINCLUDE := -I $(TESTDIR)
TESTOBJS := $(TESTSRC:$(TESTDIR)/%.cpp=$(OBJDIR)/%.o)
TESTTARGETS := $(TESTSRC:$(TESTDIR)/%.cpp=$(TESTBINDIR)/%)
TESTSPY := $(wildcard $(TESTDIR)/*.py)

tests: $(TESTTARGETS)

$(TESTOBJS):  $(OBJDIR)/%.o : $(TESTDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TESTINCLUDE) -c $^ -o $@

$(TESTTARGETS): $(TESTBINDIR)/% : $(OBJDIR)/%.o $(SUBOBJS) $(EXTOBJS)
	$(CXX) $(CXXFLAGS) $^  $(LIBS) -o $@

### OLD tests
derivCovar: derivCovar.o $(GBUTIL_OBJ)/BinomFact.o
	$(CXX) $(CXXFLAGS) $^  $(LIBS) -o $@

test: test.o 
	$(CXX) $(CXXFLAGS) $^  $(LIBS) -o $@

testMultiGauss: obj/testMultiGauss.o obj/Distributions.o
	$(CXX) $(CXXFLAGS) $^  $(LIBS) -o $@

testGaussMoments: obj/testGaussMoments.o obj/Distributions.o obj/GaussMoments.o $(EXTOBJS)
	$(CXX) $(CXXFLAGS) $^  $(LIBS) -o $@

testGaussianGalaxy: obj/testGaussianGalaxy.o obj/Galaxy.o obj/Distributions.o obj/Moment.o obj/GaussMoments.o $(EXTOBJS)
	$(CXX) $(CXXFLAGS) $^  $(LIBS) -o $@

###############################################################
## Standard stuff:
###############################################################

exts:
	for dir in $(EXTDIRS); do (cd $$dir && $(MAKE)); done

depend: local-depend
	for dir in $(EXTDIRS); do (cd $$dir && $(MAKE) depend); done
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MM $(EXECS) $(SUBS) > .$@

local-depend:
	$(RM) .depend
	for src in $(SUBS:%.cpp=%) $(EXECS:%.cpp=%); \
	   do $(CXX) $(CXXFLAGS) $(INCLUDES) -MM $$src.cpp -MT obj/$$src.o >> .depend; \
        done

clean: local-clean
	for dir in $(EXTDIRS); do (cd $$dir && $(MAKE) clean); done

local-clean:
	rm -f $(OBJDIR)/*.o $(BINDIR)/* $(TESTBINDIR)/* *~ *.dvi *.aux core .depend

ifeq (.depend, $(wildcard .depend))
include .depend
endif

.PHONY: all install dist depend clean 
