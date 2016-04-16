#include "viewerParameter.h"
#include <iostream>
#include <boost/lexical_cast.hpp>
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/filesystem/convenience.hpp"
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/case_conv.hpp>

namespace jul {

ViewerParam::ViewerParam(int argc, char *argv[])
{
	m_opt = kUnknown;
	
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
			
		if(strcmp(argv[i], "-tv") == 0 || strcmp(argv[i], "--testVoxel") == 0) {
			m_opt = kTestVoxel;
			break;
		}
		
		if(strcmp(argv[i], "-ta") == 0 || strcmp(argv[i], "--testAsset") == 0) {
			if(i==argc-1) {
				std::cout<<"\n --testAsset value is not set";
				m_opt = kUnknown;
				break;
			}
			m_inFileName = argv[i+1];
			m_opt = kTestAsset;
		}
	}
	
}

ViewerParam::~ViewerParam() {}

bool ViewerParam::isValid() const
{ return m_opt > 0; }

ViewerParam::OperationFlag ViewerParam::operation() const
{ return m_opt; }

void ViewerParam::PrintHelp()
{
	std::cout<<"\n julia (Voxel Render Research) version 20160402"
	<<"\nDescription:\n large data set test."
	<<"\nUsage:\n julia [option]\nOptions:"
	<<"\n -tv or --testVoxel    prototype to draw voxel with contours"
	<<"\n -ta or --testAsset    string1    filename of asset"
	<<"\n     prototype to draw triangle asset"
	<<"\n -h or --help    print this information"
	<<std::endl;
}

const std::string & ViewerParam::inFileName() const
{ return m_inFileName; }

std::string ViewerParam::operationTitle() const
{
	if(m_opt == kTestVoxel) return "Voxel Contour Ray-Cast";
	if(m_opt == kTestAsset) return "Triangle Asset " + inFileName();
	
	return "unknown";
}

}
