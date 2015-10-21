/*
 *  H5IO.cpp
 *  opium
 *
 *  Created by jian zhang on 10/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "H5IO.h"
#include <SHelper.h>

std::string H5IO::BeheadName("");

void H5IO::CreateGroup(const std::string & name)
{
	std::vector<std::string> allNames; 
    SHelper::listAllNames(name, allNames);
    std::vector<std::string>::const_iterator it = allNames.begin();
    for(;it!=allNames.end();++it) {
        HBase grpPar(*it);
        grpPar.close();
    }
	allNames.clear();
}
