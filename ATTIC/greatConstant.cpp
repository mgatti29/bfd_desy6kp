// Estimate constant shear from a FITS binary table of Pqr's
#include <fstream>

#include "StringStuff.h"
#include "Pqr.h"
#include "FitsTable.h"
#include "FitsImage.h"
#include "MomentIndices.h"

string usage = "greatConstant: Estimate constant shear on a field using PQR info\n"
  "  saved into a FITS binary table\n"
  "usage: greatConstant <moment file> [minUnique=1]\n"
  "     moment file: name of FITS file with binary table in extension 1\n"
  "     minUnique: do not use galaxies using < this many template images\n"
  "stdout: total PQR and shear estimate";

using namespace bfd;

int main(int argc,
	 char *argv[])
{
  const int UseMoments=USE_ALL_MOMENTS;
  typedef MomentIndices<UseMoments> MI;

  if (argc<2 || argc>3) {
    cerr << usage << endl;
    exit(1);
  }
  string momentFile = argv[1];
  int minUnique = argc>2 ? atoi(argv[2]) : 1;
  try {

    cout << "#" << stringstuff::taggedCommandLine(argc,argv) << endl;
    FITS::FitsTable ft(momentFile, FITS::ReadOnly, 1);
    img::FTable tab = ft.use();

    const int nTarget = tab.nrows();
    int useN = 0;
    int lowN = 0;
    int deselectN = 0;
    vector<float> vp(6);
    int nUnique;
    Pqr accumulator;
    for (int i=0; i<nTarget; i++) {
      tab.readCell(vp, "PQR", i);
      tab.readCell(nUnique, "NUNIQUE", i);
      Pqr linearPqr;
      for (int j=0; j<Pqr::SIZE; j++) linearPqr[j]=vp[j];
      if (nUnique < 0) {
	deselectN++;
	accumulator += linearPqr.neglog();
      } else if (nUnique<minUnique) {
	lowN++;
      } else {
	accumulator += linearPqr.neglog();
	useN++;
      }
    } // end target loop

    // Output results:
    cout << "Use " << useN << " deselect " << deselectN << " reject " << lowN << endl;
    cout << "PQR: " << accumulator[Pqr::P] 
	 << " " << accumulator[Pqr::DG1]
	 << " " << accumulator[Pqr::DG2]
	 << " " << accumulator[Pqr::D2G1G1]
	 << " " << accumulator[Pqr::D2G2G2]
	 << " " << accumulator[Pqr::D2G1G2]
	 << endl;
    Pqr::GVector gMean;
    Pqr::GMatrix gCov;
    accumulator.getG(gMean, gCov);
    cout << "g1: " << gMean[Pqr::G1] << " +- " << sqrt(gCov(Pqr::G1,Pqr::G1))
	 << " g2: " << gMean[Pqr::G2] << " +- " << sqrt(gCov(Pqr::G2,Pqr::G2))
	 << " r: " << gCov(Pqr::G1,Pqr::G2) / sqrt(gCov(Pqr::G2,Pqr::G2)*gCov(Pqr::G1,Pqr::G1))
	 << endl;

    // Potential Great3-format output:
    cout << ">> " << momentFile << " " << gMean[Pqr::G1] << " " << gMean[Pqr::G2] << endl;

  } catch (std::runtime_error& e) {
    quit(e,1);
  }
}


