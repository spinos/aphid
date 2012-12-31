#pragma once
#include <Alembic/AbcGeom/IPolyMesh.h>
class ALIMesh {
public:
	ALIMesh(Alembic::AbcGeom::IPolyMesh &obj);
	~ALIMesh();
	
	/*void addP(const float *vertices, const unsigned &numVertices);
	void addFaceConnection(const unsigned *indices, const unsigned &numIndices);
	void addFaceCount(const unsigned *counts, const unsigned &numCounts);
	void addUV(const float *uvs, const unsigned &numUVs, const unsigned *indices, const unsigned &numIds);
	bool isTopologyValid();
	void write();*/
	void verbose();
private:
	Alembic::AbcGeom::IPolyMeshSchema m_schema;
	Alembic::AbcGeom::IPolyMeshSchema::Sample m_sample;
};
