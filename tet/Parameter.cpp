#include "Parameter.h"

#include <boost/lexical_cast.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#define _WIN32
#include <ExrImage.h>
#include <iostream>
// #include <OpenEXR/ImathLimits.h>

namespace ttg {

Parameter::Parameter(int argc, char *argv[])
{
	m_operation = kHelp;
	
	if(argc < 2) {
		std::cout<<"\n tet (Tetrahedral Mesh Generation Research) version 20160601"
			<<std::endl;
	}
	else {
		int i = 1;
		for(;i<argc;++i) {
			if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
				m_operation = kHelp;
			}
			if(strcmp(argv[i], "-h2") == 0 || strcmp(argv[i], "--hilbert2d") == 0) {
				m_operation = kHilbert2D;
			}
			if(strcmp(argv[i], "-h3") == 0 || strcmp(argv[i], "--hilbert3d") == 0) {
				m_operation = kHilbert3D;
			}
			if(strcmp(argv[i], "-d2") == 0 || strcmp(argv[i], "--delauney2d") == 0) {
				m_operation = kDelaunay2D;
			}
			if(strcmp(argv[i], "-d3") == 0 || strcmp(argv[i], "--delauney3d") == 0) {
				m_operation = kDelaunay3D;
			}
			if(strcmp(argv[i], "-b3") == 0 || strcmp(argv[i], "--bcc3d") == 0) {
				m_operation = kBcc3D;
			}
			if(strcmp(argv[i], "-sf") == 0 || strcmp(argv[i], "--superformula") == 0) {
				m_operation = kSuperformula;
			}
			if(strcmp(argv[i], "-sfp") == 0 || strcmp(argv[i], "--superformulaPoissonDisk") == 0) {
				m_operation = kSuperformulaPoissonDisk;
			}
			if(strcmp(argv[i], "-bt") == 0 || strcmp(argv[i], "--bccTetrahedralize") == 0) {
				m_operation = kBccTetrahedralize;
			}
			if(strcmp(argv[i], "-df") == 0 || strcmp(argv[i], "--distanceField") == 0) {
				m_operation = kDistanceField;
			}
			if(strcmp(argv[i], "-rbr") == 0 || strcmp(argv[i], "--redBlueRefine") == 0) {
				m_operation = kRedblueRefine;
			}
			if(strcmp(argv[i], "-ft") == 0 || strcmp(argv[i], "--fieldTetrahedralize") == 0) {
				m_operation = kFieldTetrahedralize;
			}
			if(strcmp(argv[i], "-ag") == 0 || strcmp(argv[i], "--adaptiveGrid") == 0) {
				m_operation = kAaptiveGrid;
			}
			if(strcmp(argv[i], "-kd") == 0 || strcmp(argv[i], "--kdDistance") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --kdDistance value is not set";
					m_operation = kHelp;
					break;
				}
				m_inFileName = argv[i+1];
				m_operation = kKdistance;
			}
			if(strcmp(argv[i], "-vd") == 0 || strcmp(argv[i], "--viewDependent") == 0) {
				m_operation = kViewDependentGrid;
			}
			if(strcmp(argv[i], "-n3") == 0 || strcmp(argv[i], "--noise3") == 0) {
				m_operation = kNoise3;
			}
			if(strcmp(argv[i], "-lg") == 0 || strcmp(argv[i], "--legendre") == 0) {
				m_operation = kLegendre;
			}
			if(strcmp(argv[i], "-l2") == 0 || strcmp(argv[i], "--legendre2D") == 0) {
				m_operation = kLegendre2D;
			}
			if(strcmp(argv[i], "-l3") == 0 || strcmp(argv[i], "--legendre3D") == 0) {
				m_operation = kLegendre3D;
			}
			if(strcmp(argv[i], "-is") == 0 || strcmp(argv[i], "--intersect") == 0) {
				m_operation = kIntersect;
			}
		}
	}
	
}

Parameter::~Parameter() {}

Parameter::Operation Parameter::operation() const
{ return m_operation; }

const std::string & Parameter::inFileName() const
{ return m_inFileName; }

const std::string & Parameter::outFileName() const
{ return m_outFileName; }

void Parameter::PrintHelp()
{
	std::cout<<"\n tet (Tetrahedral Mesh Generation Research) version 20160601"
	<<"\nUsage:\n tet [option] [file]"
	<<"\nDescription:\n generates tetrahedral mesh for FEM"
	<<"\nOptions:\n -h or --help    print this information"
	<<"\n -h2 or --hilbert2d    test 2D hilbert curve"
	<<"\n -h3 or --hilbert3d    test 3D hilbert curve"
	<<"\n -d2 or --delauney2d    test 2D delauney"
	<<"\n -d3 or --delauney3d    test 3D delauney"
	<<"\n -b3 or --bcc3d    test 3D BCC"
	<<"\n -sf or --superformula    test 3D superformula"
	<<"\n -sfp or --superformulaPoissonDisk    test 3D superformula + Poisson Disk Sampling"
	<<"\n -bt or --bccTetrahedralize    test 3D superformula + Poisson Disk Sampling + BCC"
	<<"\n -df or --distanceField    test 3D distance transform on BCC grid"
	<<"\n -rbr or --redBlueRefine    test red blue refine of a tetrahedron"
	<<"\n -ft or --fieldTetrahedralize    test tetrahedralize based on signed distance field"
	<<"\n -ag or --adaptiveGrid    test adaptive grid"
	<<"\n -kd or --kdDistance    input_filename    test kd-tree distance field + adaptive grid"
	<<"\n -vd or --viewDependent    test view dependent grid"
	<<"\n -n3 or --noise3    test 3d noise"
	<<"\n -lg or --legendre    legendre polynomial approximation"
	<<"\n -l2 or --legendre2D    2D legendre polynomial approximation"
	<<"\n -l3 or --legendre3D    3D legendre polynomial approximation"
	<<"\n -is or --intersect    intersect test"
	<<"\nHot keys:"
	<<"\n m/n    progress forward/backward"
	<<"\n";
}

}
