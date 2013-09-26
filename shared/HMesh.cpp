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
#include <SHelper.h>
HMesh::HMesh() {}
HMesh::~HMesh() {}

char HMesh::verifyType(HObject & grp)
{
	if(!hasNamedAttr(grp, ".polyc"))
		return 0;

	if(!hasNamedAttr(grp, ".polyv"))
		return 0;
	
	if(!hasNamedAttr(grp, ".p"))
		return 0;
	
	return 1;
}

char HMesh::save()
{
	HGroup grpMesh(getName());
	if(!grpMesh.open()) {
		std::cout<<"cannot open group "<<getName();
		return 0;
	}
	
	int nv = _numVertices;
	addIntAttr("/.nv", &nv);
	
	int nf = m_numPolygons;
	addIntAttr("/.nf", &nf);
	
	int nfv = m_numPolygonVertices;
	addIntAttr("/.nfv", &nfv);
	
	addVector3Data("/.p", nv, vertices());
	addIntData("/.polyc", m_numPolygons, (int *)m_polygonCounts);
	addIntData("/.polyv", m_numPolygonVertices, (int *)m_polygonIndices);
	
	grpMesh.close();

	return 1;
}

char HMesh::load()
{
	HGroup chd(getName());
	if(!chd.open()) {
		std::cout<<"cannot open group "<<getName();
		return 0;
	}
	
	std::stringstream sst;
	
	int numVertices = 3;
	
	readIntAttr("/.nv", &numVertices);
	
	createVertices(numVertices);
	
	int numPolygons = 1;
	
	readIntAttr("/.nf", &numPolygons);
	
	createPolygonCounts(numPolygons);
	
	int numPolygonVertices = 3;
	
	readIntAttr("/.nfv", &numPolygonVertices);
	
	createPolygonIndices(numPolygonVertices);
	
	readVector3Data("/.p", numVertices, vertices());
	readIntData("/.polyc", numPolygons, m_polygonCounts);
	readIntData("/.polyv", numPolygonVertices, m_polygonIndices);
	
	chd.close();
	
	processTriangleFromPolygon();
	processQuadFromPolygon();
	processRealEdgeFromPolygon();
	
	return 1;
}
//:~
