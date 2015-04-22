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
#include <CurveGroup.h>
#include <BaseBuffer.h>

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
	
	return 1;
}

char HTetrahedronMesh::save(TetrahedronMeshData * tetra)
{
	int nv = tetra->m_numPoints;
	if(!hasNamedAttr(".nv"))
		addIntAttr(".nv");
	
	writeIntAttr(".nv", &nv);
	
	int nt = tetra->m_numTetrahedrons;
	if(!hasNamedAttr(".nt"))
		addIntAttr(".nt");
	
	writeIntAttr(".nt", &nt);
	
	if(!hasNamedData(".p"))
	    addVector3Data(".p", nv);
	
	writeVector3Data(".p", nv, (Vector3F *)tetra->m_pointBuf->data());
	
	if(!hasNamedData(".a"))
	    addIntData(".a", nv);
	
	writeIntData(".a", nv, (int *)tetra->m_anchorBuf->data());
		
	if(!hasNamedData(".v"))
	    addIntData(".v", nt * 4);
	
	writeIntData(".v", nt * 4, (int *)tetra->m_indexBuf->data());

	return 1;
}

char HTetrahedronMesh::load(TetrahedronMeshData * tetra)
{
	if(!verifyType()) return false;
	int nv = 4;
	
	readIntAttr(".nv", &nv);
	
	int nt = 1;
	
	readIntAttr(".nt", &nt);
	
	tetra->m_numTetrahedrons = nt;
    tetra->m_numPoints = nv;
	tetra->m_anchorBuf = new BaseBuffer;
	tetra->m_anchorBuf->create(nv * 4);
	tetra->m_pointBuf = new BaseBuffer;
	tetra->m_pointBuf->create(nv * 12);
	tetra->m_indexBuf = new BaseBuffer;
	tetra->m_indexBuf->create(nt * 4 * 4);
	
	readVector3Data(".p", nv, (Vector3F *)tetra->m_pointBuf->data());
	readIntData(".a", nv, (unsigned *)tetra->m_anchorBuf->data());
	readIntData(".v", nt * 4, (unsigned *)tetra->m_indexBuf->data());
	
	return 1;
}
//:~