/*
 *  HTriangleMeshGroup.cpp
 *  aphid
 *
 *  Created by jian zhang on 7/6/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "HTriangleMeshGroup.h"
#include "ATriangleMeshGroup.h"

namespace aphid {

HTriangleMeshGroup::HTriangleMeshGroup(const std::string & path) : HTriangleMesh(path) {}
HTriangleMeshGroup::~HTriangleMeshGroup() {}

char HTriangleMeshGroup::verifyType()
{
	if(!hasNamedAttr(".npart"))
		return 0;
		
	return HTriangleMesh::verifyType();
}

char HTriangleMeshGroup::save(ATriangleMeshGroup * tri)
{
	if(!hasNamedAttr(".npart"))
		addIntAttr(".npart");
		
	int np = tri->numStripes();
	writeIntAttr(".npart", &np);
	
	if(!hasNamedData(".pntdrift"))
		addIntData(".pntdrift", np);
		
	writeIntData(".pntdrift", np, (int *)tri->pointDrifts());
	
	if(!hasNamedData(".inddrift"))
		addIntData(".inddrift", np);
		
	writeIntData(".inddrift", np, (int *)tri->indexDrifts());
	
	return HTriangleMesh::save(tri);
}

char HTriangleMeshGroup::load(ATriangleMeshGroup * tri)
{
	int npart = 1;
	readIntAttr(".npart", &npart);
	
	int nv = 3;
	readIntAttr(".nv", &nv);
	
	int nt = 1;
	readIntAttr(".ntri", &nt);
	
	tri->create(nv, nt, npart);
	
	readIntData(".pntdrift", npart, (unsigned *)tri->pointDrifts());
	readIntData(".inddrift", npart, (unsigned *)tri->indexDrifts());
	
	return HTriangleMesh::readAftCreation(tri);
}

}