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

namespace aphid {

std::string H5IO::BeheadName("");

void H5IO::CreateGroup(const std::string & name)
{
    std::cout<<"\n h5 io create group "<<name
    <<" in file "<<HObject::FileIO.fileName();
/// hierarchy to name
	std::vector<std::string> allNames; 
    SHelper::listAllNames(name, allNames);
    std::vector<std::string>::const_iterator it = allNames.begin();
    for(;it!=allNames.end();++it) {
        HBase grpPar(*it);
        grpPar.close();
    }
	allNames.clear();
}

void H5IO::H5PathName(std::string & dst)
{
// behead and strip ns
	if(BeheadName.size() > 1) SHelper::behead(dst, BeheadName);
    SHelper::removeAnyNamespace(dst);
}

}