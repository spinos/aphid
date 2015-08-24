/*
 *  BlockBccMeshBuilder.h
 *  bcc
 *
 *  Created by jian zhang on 8/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <AOrientedBox.h>
#include <deque>
class CartesianGrid;
class ATetrahedronMesh;
class BlockBccMeshBuilder {
public:
	BlockBccMeshBuilder();
	virtual ~BlockBccMeshBuilder();
	
	void build(const AOrientedBox & ob, 
				int gx, int gy, int gz,
				unsigned & numVertices, unsigned & numTetrahedrons);
				
	void getResult(ATetrahedronMesh * mesh);
protected:
    void addTetrahedron(Vector3F * v, unsigned * ind);
private:
	void addNorth(const Vector3F & center, float size, float hsize);
	void addEast(const Vector3F & center, float size, float hsize, int i, int n);
	void addDepth(const Vector3F & center, float size, float hsize, int i, int n);
private:
    CartesianGrid * m_verticesPool;
	std::deque<unsigned > m_indices;
};