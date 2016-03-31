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
	m_opt = kUnknown;
	bool foundImages = false;
	m_cellSize = 128;
	
	if(argc < 2) {
		std::cout<<"\n too few arguments "<<argc;
		return;
	}
	
	int i = 1;
	for(;i<argc;++i) {
		if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
				m_opt = kHelp;
				break;
			}
			
		if(strcmp(argv[i], "-init") == 0 || strcmp(argv[i], "--initialize") == 0) {
			if(i==argc-1) {
				std::cout<<"\n --initialize value is not set";
				m_opt = kUnknown;
				break;
			}
			m_outFileName = argv[i+1];
			m_opt = kInitialize;
		}
		
		if(strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--insert") == 0) {
			if(i==argc-2) {
				std::cout<<"\n --insert value is not set";
				m_opt = kUnknown;
				break;
			}
			m_inFileName = argv[i+1];
			m_outFileName = argv[i+2];
			m_opt = kInsert;
		}
		
		if(strcmp(argv[i], "-cz") == 0 || strcmp(argv[i], "--cellSize") == 0) {
			if(i==argc-1) {
				std::cout<<"\n --cellSize value is not set";
				m_opt = kUnknown;
				break;
			}
			try {
				int cz = boost::lexical_cast<int>(argv[i+1]);
				if(cz < 8) {
					std::cout<<"\n bad --cellSize value (< 8)";
					m_opt = kUnknown;
					break;
				}
				m_cellSize = cz;
			} catch(const boost::bad_lexical_cast &) {
				std::cout<<"\n bad --cellSize value "<<argv[i+1];
				m_opt = kUnknown;
				break;
			}
		}
			
			if(strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--generate") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --generate value is not set";
					m_opt = kUnknown;
					break;
				}
				m_outFileName = argv[i+1];
				m_opt = kGenerate;
			}
			
			if(strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tree") == 0) {
				if(i==argc-1) {
					std::cout<<"\n --tree value is not set";
					m_opt = kUnknown;
					break;
				}
				m_outFileName = argv[i+1];
				m_opt = kBuildTree;
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
	std::cout<<"\n julia (Voxel Render Research) version 20160331"
	<<"\nDescription:\n large data set test."
	<<"\nUsage:\n julia [option]\nOptions:"
	<<"\n -init or --initialize    string    filename of output file storing the wrold"
	<<"\n -cz or --cellSize    int    size of world grid cell, default 128"
	<<"\n                             must >= 8, only works with -init"
	<<"\n -i or --insert    string1    string2    insert assect from file1 into file2"
	//<<"\n -g or --generate    string    filename of output file storing the data"
	//<<"\n -t or --tree    string    filename of output file storing the data to build ntree"
	<<"\n -h or --help    print this information"
	<<"\n";
}

const std::string & Parameter::inFileName() const
{ return m_inFileName; }

const std::string & Parameter::outFileName() const
{ return m_outFileName; }

const int & Parameter::cellSize() const
{ return m_cellSize; }

}
