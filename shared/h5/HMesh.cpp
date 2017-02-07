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

namespace aphid {

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
		
	int nfv = mesh->getNumPolygonFaceVertices();
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
	
	writeVector3Data(".p", nv, mesh->getVertices());
		
	if(!hasNamedData(".polyc"))
	    addIntData(".polyc", nf);
	
	writeIntData(".polyc", nf, (int *)mesh->getPolygonCounts());
	
	if(!hasNamedData(".polyv"))
	    addIntData(".polyv", nfv);
	
	std::cout<<" polyv[0]"<<mesh->getPolygonIndices()[0]<<"\n";
	std::cout<<" polyv["<<nfv<<"-1]"<<mesh->getPolygonIndices()[nfv - 1]<<"\n";
	writeIntData(".polyv", nfv, (int *)mesh->getPolygonIndices());
	
	if(!hasNamedData(".us"))
		addFloatData(".us", nuv);
		
	writeFloatData(".us", nuv, mesh->getUs());
	
	if(!hasNamedData(".vs"))
		addFloatData(".vs", nuv);
		
	writeFloatData(".vs", nuv, mesh->getVs());
	
	if(!hasNamedData(".uvids"))
		addIntData(".uvids", nuvid);
		
	writeIntData(".uvids", nuvid, (int *)mesh->getUvIds());

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
	
	mesh->createPolygonUV(numUVs, numUVIds);
	
	readFloatData(".us", numUVs, mesh->us());
	readFloatData(".vs", numUVs, mesh->vs());
	readIntData(".uvids", numUVIds, mesh->uvIds());

	mesh->processTriangleFromPolygon();
	mesh->processQuadFromPolygon();
	
	mesh->verbose();
	
	return 1;
}

char HMesh::saveFaceTag(BaseMesh * mesh, const std::string & tagName, const std::string & dataName)
{
	if(!hasNamedData(dataName.c_str()))
		addCharData(dataName.c_str(), mesh->getNumFaces());
	
	std::cout<<"write face tag"<<tagName;
	writeCharData(dataName.c_str(),  mesh->getNumFaces(), mesh->perFaceTag(tagName));
	return 1;
}

char HMesh::loadFaceTag(BaseMesh * mesh, const std::string & tagName, const std::string & dataName)
{
	char * g = mesh->perFaceTag(tagName);
	if(hasNamedData(dataName.c_str())) {
		readCharData(dataName.c_str(),  mesh->getNumFaces(), g);
	}
	else {
		std::cout<<"WARNING: reset face tag "<<tagName;
		for(unsigned i =0; i < mesh->getNumFaces(); i++) g[i] = 1;
	}
	return 1;
}

}
//:~
