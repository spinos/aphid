/*
 *  ALMesh.h
 *  helloAbc
 *
 *  Created by jian zhang on 11/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Alembic/AbcGeom/OPolyMesh.h>
class ALMesh {
public:
	ALMesh(Alembic::AbcGeom::OPolyMesh &obj);
	~ALMesh();
	
	void addP(const float *vertices, const unsigned &numVertices);
	void addFaceConnection(const unsigned *indices, const unsigned &numIndices);
	void addFaceCount(const unsigned *counts, const unsigned &numCounts);
	void addUV(const float *uvs, const unsigned &numUVs, const unsigned *indices, const unsigned &numIds);
	bool isTopologyValid();
	void write();
private:
	Alembic::AbcGeom::OPolyMeshSchema m_schema;
	Alembic::AbcGeom::OPolyMeshSchema::Sample m_sample;
};