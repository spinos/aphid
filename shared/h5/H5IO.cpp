/*
 *  H5IO.cpp
 *  opium
 *
 *  Created by jian zhang on 10/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "H5IO.h"
#include <foundation/SHelper.h>

namespace aphid {

std::string H5IO::BeheadName("");

bool H5IO::begin(const std::string & filename,
					HDocument::OpenMode om)
{
	if(!HObject::FileIO.open(filename.c_str(), om) ) {
		if(om == HDocument::oCreate) 
			std::cout<<"\n  h5io cannot create file "<<filename;
		else if(om == HDocument::oReadAndWrite) 
			std::cout<<"\n  h5io cannot read/write file "<<filename;
		else 
			std::cout<<"\n  h5io cannot read file "<<filename;
		return false;
	}
	m_doc = HObject::FileIO;
	std::cout<<"\n h5io open file "<<m_doc.fileName()<<"\n";
	return true;
}

void H5IO::end()
{
	std::cout<<"\n h5io close file "<<m_doc.fileName()<<"\n";
	m_doc.close();
	//HObject::FileIO.close();
}

bool H5IO::objectExists(const std::string & fullPath)
{
	std::vector<std::string> allNames; 
    SHelper::Split(fullPath, allNames);
	bool stat = true;
	HBase rt("/");
	std::vector<HBase *> openedGrps;
    openedGrps.push_back(&rt);
	
    std::vector<std::string>::const_iterator it = allNames.begin();
    for(;it!=allNames.end();++it) {
		stat = openedGrps.back()->hasNamedChild((*it).c_str() );
		
		if(stat ) {
			openedGrps.push_back(new HBase(openedGrps.back()->childPath(*it ) ) );
			
		}
		else {
			std::cout<<"\n  h5io cannot find "<<openedGrps.back()->childPath(*it )
					<<" in file "<<HObject::FileIO.fileName();
			std::cout.flush();
			stat = false;
			break;
		}
    }
	
	std::vector<HBase *>::iterator cit = openedGrps.begin();
    for(;cit!=openedGrps.end();++cit) {
		(*cit)->close();
	}
	
	return stat;
}

void H5IO::CreateGroup(const std::string & name)
{
    //std::cout<<"\n h5 io create group "<<name
    //<<" in file "<<HObject::FileIO.fileName();
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

HBase * H5IO::GetH5dHeadGroup()
{
    if(BeheadName.size() < 2) {
        return new HBase("/");
    }
    
    if(!objectExists(BeheadName)) {
        return new HBase("/");
   }
   
   std::cout<<"\n h5 head name is "<<BeheadName;
   return new HBase(BeheadName);
}


}
