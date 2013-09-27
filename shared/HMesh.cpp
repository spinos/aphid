/*
 *  HMesh.cpp
 *  masqmaya
 *
 *  Created by jian zhang on 4/13/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "HMesh.h"
#include <AllHdf.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <BaseMesh.h>
HMesh::HMesh(const std::string & path) : HBase(path) 
{
}

HMesh::~HMesh() {}

char HMesh::verifyType()
{
	if(!hasNamedAttr(".polyc"))
		return 0;

	if(!hasNamedAttr(".polyv"))
		return 0;
	
	if(!hasNamedAttr(".p"))
		return 0;
	
	return 1;
}

char HMesh::save(BaseMesh * mesh)
{
	mesh->verbose();
	
	int nv = mesh->getNumVertices();
	if(!hasNamedAttr(".nv"))
		addIntAttr(".nv");
	
	writeIntAttr(".nv", &nv);
	
	int nf = mesh->getNumPolygons();
	if(!hasNamedAttr(".nf"))
		addIntAttr(".nf");
	
	writeIntAttr(".nf", &nf);
		
	int nfv = mesh->getNumFaceVertices();
	if(!hasNamedAttr(".nfv"))
		addIntAttr(".nfv");
	
	writeIntAttr(".nfv", &nfv);
	
	int nuv = mesh->getNumUVs();
	if(!hasNamedAttr(".nuv"))
		addIntAttr(".nuv");
	
	writeIntAttr(".nuv", &nuv);
	
	int nuvid = mesh->getNumUVIds();
	if(!hasNamedAttr(".nuvid"))
		addIntAttr(".nuvid");
	
	writeIntAttr(".nuvid", &nuvid);
	
	if(!hasNamedData(".p"))
	    addVector3Data(".p", nv);
	
	writeVector3Data(".p", nv, mesh->vertices());
		
	if(!hasNamedData(".polyc"))
	    addIntData(".polyc", nf);
	
	writeIntData(".polyc", nf, (int *)mesh->polygonCounts());
	
	if(!hasNamedData(".polyv"))
	    addIntData(".polyv", nfv);
	
	writeIntData(".polyv", nfv, (int *)mesh->polygonIndices());

	return 1;
}

char HMesh::load(BaseMesh * mesh)
{
	int numVertices = 3;
	
	readIntAttr(".nv", &numVertices);
	
	int numPolygons = 1;
	
	readIntAttr(".nf", &numPolygons);
	
	int numPolygonVertices = 3;
	
	readIntAttr(".nfv", &numPolygonVertices);
	
	int numUVs = 3;
	readIntAttr(".nuv", &numUVs);
	
	int numUVIds = 3;
	readIntAttr(".nuvid", &numUVIds);
	
	mesh->createVertices(numVertices);
	mesh->createPolygonCounts(numPolygons);
	mesh->createPolygonIndices(numPolygonVertices);
	
	readVector3Data(".p", numVertices, mesh->vertices());
	readIntData(".polyc", numPolygons, mesh->polygonCounts());
	readIntData(".polyv", numPolygonVertices, mesh->polygonIndices());
	
	mesh->processTriangleFromPolygon();
	mesh->processQuadFromPolygon();
	
	mesh->createPolygonUV(numUVs, numUVIds);
	/*
	
	for(unsigned i = 0; i < esm->getNumUVs(); i++) {
		mesh->us()[i] = esm->getUs()[i];
		mesh->vs()[i] = esm->getVs()[i];
	}
	
	for(unsigned i = 0; i < esm->getNumUVIds(); i++)
		mesh->uvIds()[i] = esm->getUVIds()[i];
*/
	mesh->verbose();
	
	return 1;
}
//:~
