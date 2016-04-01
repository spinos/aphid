/*
 *  NTreeIO.cpp
 *  
 *
 *  Created by jian zhang on 3/10/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "NTreeIO.h"
#include <HWorldGrid.h>
#include <HInnerGrid.h>

namespace aphid {

NTreeIO::NTreeIO() 
{}

bool NTreeIO::begin(const std::string & filename,
					HDocument::OpenMode om)
{
	if(!HObject::FileIO.open(filename.c_str(), om) ) {
		if(om == HDocument::oCreate) 
			std::cout<<"\n NTree IO cannot create file "<<filename;
		else if(om == HDocument::oReadAndWrite) 
			std::cout<<"\n NTree IO cannot read/write file "<<filename;
		else 
			std::cout<<"\n NTree IO cannot read file "<<filename;
		return false;
	}
	m_doc = HObject::FileIO;
	std::cout<<"\n ntree io open file "<<m_doc.fileName()<<"\n";
	return true;
}

void NTreeIO::end()
{
	std::cout<<"\n ntree io close file "<<m_doc.fileName()<<"\n";
	m_doc.close();
	//HObject::FileIO.close();
}

bool NTreeIO::findGrid(std::string & name, const std::string & grpName)
{
	std::vector<std::string > gridNames;
	HBase r(grpName);
	r.lsTypedChild<sdb::HVarGrid>(gridNames);
	r.close();
	
	if(gridNames.size() <1) {
		std::cout<<"\n found no grid";
		return false;
	}
	name = gridNames[0];
	return true;
}

bool NTreeIO::findTree(std::string & name,
				const std::string & grpName)
{
	std::vector<std::string > treeNames;
	HBase r(grpName);
	r.lsTypedChild<HBaseNTree>(treeNames);
	r.close();
	
	if(treeNames.size() <1) {
		std::cout<<"\n found no tree";
		return false;
	}
	name = treeNames[0];
	return true;
}

cvx::ShapeType NTreeIO::gridValueType(const std::string & name)
{
	cvx::ShapeType vt = cvx::TUnknown;
    
    sdb::HVarGrid vg(name);
    vg.load();
    std::cout<<"\n value type ";
    switch(vg.valueType() ) {
        case cvx::TSphere :
        std::cout<<"sphere";
        vt = cvx::TSphere;
        break;
    default:
        std::cout<<"unsupported";
        break;
    }
    vg.close();
	
	return vt;
}

}