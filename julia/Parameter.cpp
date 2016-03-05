#include "Parameter.h"
#include <iostream>
#include <boost/lexical_cast.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/case_conv.hpp>

namespace jul {

Parameter::Parameter(int argc, char *argv[])
{
	std::cout<<"\n lfr (Julia Set) version 20160101";

	m_opt = kUnknown;
	
	bool foundImages = false;
	if(argc == 1) {
		
	}
	else {
		int i = 1;
		for(;i<argc;++i) {
			if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
				m_opt = kHelp;
			}
			
			if(strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--generate") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --generate value is not set";
					break;
				}
				m_outFileName = argv[i+1];
				m_opt = kGenerate;
			}
			
			if(strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tree") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --tree value is not set";
					break;
				}
				m_outFileName = argv[i+1];
				m_opt = kBuildTree;
			}
		}
	}
}

Parameter::~Parameter() {}

bool Parameter::isValid() const
{ return m_opt > 0; }

Parameter::OperationFlag Parameter::operation() const
{ return m_opt; }

void Parameter::PrintHelp()
{
	std::cout<<"\n julia (Light Field Research) version 20151122"
	<<"\nUsage:\n julia [option]"
	<<"\nDescription:\n large data set test."
	<<"\nOptions:\n -g or --generate    string    filename of output file storing the data"
	<<"\n -t or --tree    string    filename of output file storing the data to build ntree"
	<<"\n -h or --help    print this information"
	<<"\n";
}

std::string Parameter::outFileName() const
{ return m_outFileName; }

}
