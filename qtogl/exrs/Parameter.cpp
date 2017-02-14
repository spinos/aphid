#include "Parameter.h"

#include <boost/lexical_cast.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/case_conv.hpp>

#include <iostream>

namespace exrs {

Parameter::Parameter(int argc, char *argv[])
{
	m_operation = kHelp;
	
	if(argc < 2) {
		return;
	}
	
	int i = 1;
	for(;i<argc;++i) {
		if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			m_operation = kHelp;
			break;
		}
		if(strcmp(argv[i], "-ts") == 0 || strcmp(argv[i], "--testSampler") == 0) {
			if(i==argc-1) {
				std::cout<<"\n --testSampler value is not set";
				m_operation = kHelp;
				break;
			}
			m_inFileName = argv[i+1];
			m_operation = kTestSampler;
		}
	}
	
}

Parameter::~Parameter() 
{}

Parameter::Operation Parameter::operation() const
{ return m_operation; }

const std::string & Parameter::inFileName() const
{ return m_inFileName; }

void Parameter::PrintHelp()
{
	std::cout<<"\n exrs version 20170208"
	<<"\nUsage:\n exrs [option] [file]"
	<<"\nDescription:\n EXR image test"
	<<"\nOptions:\n -h or --help    print this information"
	<<"\n -ts or --testSampler    <input_filename>    test exr image sampler"
	<<"\n";
}

}
