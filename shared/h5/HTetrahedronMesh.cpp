/*
 *  HTetrahedronMesh.cpp
 *  testbcc
 *
 *  Created by jian zhang on 4/23/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HTetrahedronMesh.h"
#include <AllHdf.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <geom/ATetrahedronMesh.h>

namespace aphid {

HTetrahedronMesh::HTetrahedronMesh(const std::string & path) : HBase(path) 
{
}

HTetrahedronMesh::~HTetrahedronMesh() {}

char HTetrahedronMesh::verifyType()
{
	if(!hasNamedAttr(".nt"))
		return 0;

	if(!hasNamedAttr(".nv"))
		return 0;
    
    if(!hasNamedAttr(".vlm"))
		return 0;
	
	return 1;
}

char HTetrahedronMesh::save(ATetrahedronMesh * tetra)
{
	int nv = tetra->numPoints();
	if(!hasNamedAttr(".nv"))
		addIntAttr(".nv");
	
	writeIntAttr(".nv", &nv);
	
	int nt = tetra->numTetrahedrons();
	if(!hasNamedAttr(".nt"))
		addIntAttr(".nt");
	
	writeIntAttr(".nt", &nt);
	
	if(!hasNamedData(".p"))
	    addVector3Data(".p", nv);
	
	writeVector3Data(".p", nv, (Vector3F *)tetra->points());
	
	if(!hasNamedData(".a"))
	    addIntData(".a", nv);
	
	writeIntData(".a", nv, (int *)tetra->anchors());
		
	if(!hasNamedData(".v"))
	    addIntData(".v", nt * 4);
	
	writeIntData(".v", nt * 4, (int *)tetra->indices());
    
    if(!hasNamedAttr(".vlm"))
	    addFloatAttr(".vlm");
    
    float vlm = tetra->volume();
    writeFloatAttr(".vlm", &vlm);

	return 1;
}

char HTetrahedronMesh::load(ATetrahedronMesh * tetra)
{
	if(!verifyType()) return false;
	int nv = 4;
	
	readIntAttr(".nv", &nv);
	
	int nt = 1;
	
	readIntAttr(".nt", &nt);
	
	tetra->create(nv, nt);

	return readAftCreation(tetra);
}

char HTetrahedronMesh::readAftCreation(ATetrahedronMesh * tetra)
{
    readVector3Data(".p", tetra->numPoints(), (Vector3F *)tetra->points());
	readIntData(".a", tetra->numPoints(), (int *)tetra->anchors());
	readIntData(".v", tetra->numTetrahedrons() * 4, (int *)tetra->indices());
	
    float vlm = 0.f;
    readFloatAttr(".vlm", &vlm);
    tetra->setVolume(vlm);
    
    return 1;
}

}
//:~