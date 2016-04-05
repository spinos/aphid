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

int NTreeIO::loadBox(sdb::VectorArray<cvx::Box> * dst, HBase * grp)
{
	if(!grp->hasNamedData(".box") ) return 0;
	HOocArray<hdata::TChar, 32, 1024> cbd(boost::str(boost::format("%1%/.box") % grp->pathToObject() ) );
	if(!cbd.openStorage(grp->fObjectId) ) return 0;
	const int nc = cbd.numCols();
	std::cout<<"\n ntreeio read "<<nc<<" box from "<<grp->pathToObject();
	cvx::Box b;
	int i=0;
	for(;i<nc;++i) {
		cbd.readColumn((char *)&b, i);
		dst->insert(b);
	}
	return dst->size();
}

int NTreeIO::loadVoxel(sdb::VectorArray<Voxel> * dst, HBase * grp)
{
	if(!grp->hasNamedData(".vox") ) return 0;
	HOocArray<hdata::TChar, 12, 1024> cbd(boost::str(boost::format("%1%/.vox") % grp->pathToObject() ) );
	if(!cbd.openStorage(grp->fObjectId) ) return 0;
	const int nc = cbd.numCols();
	std::cout<<"\n ntreeio read "<<nc<<" voxel from "<<grp->pathToObject();
	Voxel b;
	int i=0;
	for(;i<nc;++i) {
		cbd.readColumn((char *)&b, i);
		dst->insert(b);
	}
	return dst->size();
}

}