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

NTreeIO::NTreeIO() {}

bool NTreeIO::begin(const std::string & filename,
					HDocument::OpenMode om)
{
	return HObject::FileIO.open(filename.c_str(), om);
}

void NTreeIO::end()
{
	HObject::FileIO.close();
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

HNTree<cvx::Cube, KdNode4 > * NTreeIO::loadCube4Tree(const std::string & name)
{
	std::cout<<"\n cube4 tree "<<name;
	HNTree<cvx::Cube, KdNode4 > *tree = new HNTree<cvx::Cube, KdNode4 >( name );
    tree->load();
	tree->close();
	return tree;
}

void NTreeIO::loadSphereGridCoord(sdb::VectorArray<cvx::Cube> * dst, const std::string & name)
{
	sdb::HWorldGrid<sdb::HInnerGrid<hdata::TFloat, 4, 1024 >, cvx::Sphere > grd(name);
    grd.load();
    const float h = grd.gridSize();
    const float e = h * .4999f;
   cvx::Cube c;
    grd.begin();
    while(!grd.end() ) {
        c.set(grd.coordToCellCenter(grd.key() ), e);
        dst->insert(c);
        grd.next(); 
    }
	grd.close();
}

}