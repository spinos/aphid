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
	m_operation = kDelaunay2D;
	
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
		}
	}
	
}

Parameter::~Parameter() {}

Parameter::Operation Parameter::operation() const
{ return m_operation; }

void Parameter::PrintHelp()
{
	std::cout<<"\n tet (Tetrahedral Mesh Generation Research) version 20160601"
	<<"\nUsage:\n tet [option] [file]"
	<<"\nDescription:\n generates tetrahedral mesh for FEM"
	<<"\nOptions:\n -h or --help    print this information"
	<<"\n";
}

}
