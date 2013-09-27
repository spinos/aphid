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
	int nv = mesh->getNumVertices();
	if(!hasNamedAttr(".nv"))
		addIntAttr(".nv", &nv);
	else 
		writeIntAttr(".nv", &nv);
		
	int nf = mesh->getNumPolygons();
	if(!hasNamedAttr(".nf"))
		addIntAttr(".nf", &nf);
	else 
		writeIntAttr(".nf", &nf);
		
	int nfv = mesh->getNumFaceVertices();
	if(!hasNamedAttr(".nfv"))
		addIntAttr(".nfv", &nfv);
	else 
		writeIntAttr(".nfv", &nfv);
		
	std::cout<<" "<<nv<<" "<<nf<<" "<<nfv;
	std::cout<<"\np\n";
	if(!hasNamedData(".p"))
	    addVector3Data(".p", nv, mesh->vertices());
	else 
	    writeVector3Data(".p", nv, mesh->vertices());
	
	std::cout<<"\npc\n";
	if(!hasNamedData(".polyc"))
	    addIntData(".polyc", nf, (int *)mesh->polygonCounts());
	else
	    writeIntData(".polyc", nf, (int *)mesh->polygonCounts());
	
	std::cout<<"\npv\n";
	if(!hasNamedData(".polyv"))
	    addIntData(".polyv", nf, (int *)mesh->polygonIndices());
	else
	    writeIntData(".polyv", nf, (int *)mesh->polygonIndices());

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
	
	std::cout<<" "<<numVertices<<" "<<numPolygons<<" "<<numPolygonVertices;
/*
	createVertices(numVertices);
	
	
	
	createPolygonCounts(numPolygons);
	
	
	
	createPolygonIndices(numPolygonVertices);
	
	readVector3Data("/.p", numVertices, vertices());
	readIntData("/.polyc", numPolygons, m_polygonCounts);
	readIntData("/.polyv", numPolygonVertices, m_polygonIndices);
	
	processTriangleFromPolygon();
	processQuadFromPolygon();
	processRealEdgeFromPolygon();
	*/
	return 1;
}
//:~
