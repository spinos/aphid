/*
 *  ALMesh.cpp
 *  helloAbc
 *
 *  Created by jian zhang on 11/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "ALMesh.h"
#include <Alembic/AbcGeom/All.h>
using namespace Alembic::AbcGeom;
ALMesh::ALMesh(Alembic::AbcGeom::OPolyMesh &obj) 
{
	m_schema = obj.getSchema();
}

ALMesh::~ALMesh() {}

void ALMesh::addP(const float *vertices, const unsigned &numVertices)
{
	V3fArraySample p(( const V3f * )vertices, numVertices);
	m_sample.setPositions(p);
}

void ALMesh::addFaceConnection(const unsigned *indices, const unsigned &numIndices)
{
	Int32ArraySample i((Abc::int32_t *)indices, numIndices);
	m_sample.setFaceIndices(i);
}

void ALMesh::addFaceCount(const unsigned *counts, const unsigned &numCounts)
{
	Int32ArraySample i((Abc::int32_t *)counts, numCounts);
	m_sample.setFaceCounts(i);
}

void ALMesh::addUV(const float *uvs, const unsigned &numUVs, const unsigned *indices, const unsigned &numIds)
{
	Alembic::AbcGeom::OV2fGeomParam::Sample uvSamp;
	uvSamp.setScope(Alembic::AbcGeom::kFacevaryingScope);
	Alembic::AbcGeom::V2fArraySample v((const Imath::V2f *)uvs, numUVs);
	uvSamp.setVals(v);
	Alembic::Abc::UInt32ArraySample i((Abc::uint32_t *)indices, numIds);
	uvSamp.setIndices(i);
	m_sample.setUVs(uvSamp);
}

bool ALMesh::isTopologyValid()
{
	if(m_sample.getPositions().size() < 3)
		return false;
		
	if(m_sample.getFaceIndices().size() < 3)
		return false;
		
	if(m_sample.getFaceCounts().size() < 1)
		return false;
		
	return true;
}

void ALMesh::write()
{
	if(isTopologyValid())
		m_schema.set(m_sample);
}
