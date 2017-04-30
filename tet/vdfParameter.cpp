#include "Parameter.h"

#include <boost/lexical_cast.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <iostream>

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
			if(strcmp(argv[i], "-kd") == 0 || strcmp(argv[i], "--kdDistance") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --kdDistance value is not set";
					m_operation = kHelp;
					break;
				}
				m_inFileName = argv[i+1];
				m_operation = kKdistance;
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
	<<"\nDescription:\n generates distance field from mesh"
	<<"\nOptions:\n -h or --help    print this information"
	<<"\n -kd or --kdDistance    input_filename    test kd-tree distance field + adaptive grid"
	<<"\n";
}

}
