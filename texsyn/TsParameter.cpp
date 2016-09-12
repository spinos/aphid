/*
 *  TsParameter.cpp
 *  
 *
 *  Created by jian zhang on 9/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "TsParameter.h"
#include <iostream>

namespace tss {

Parameter::Parameter(int argc, char *argv[]) :
lfr::LfParameter(argc, argv)
{}

Parameter::~Parameter()
{}

void Parameter::printVersion() const
{ std::cout<<" txsn version 20160909"; }

void Parameter::printDescription() const
{
	std::cout<<"\nDescription:\n texture synthesis experiment";
}

void Parameter::printUsage() const
{
	std::cout<<"\nUsage:\n txsn [option] [file]"
	<<"\n Input file must be image of OpenEXR format. Last input can be a directory"
	<<"\n to seach for any file with name ending in .exr." 
	<<"\n If no file or directory is provided, current dir will be searched.";
}
	
}