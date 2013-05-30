/*
 *  accStencil.h
 *  catmullclark
 *
 *  Created by jian zhang on 10/30/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <AccCorner.h>
#include <AccEdge.h>
#include <AccInterior.h>
class Vector3F;
class VertexAdjacency;
class AccStencil {
public:
    
    
	AccStencil();
	~AccStencil();
	
	void setVertexPosition(Vector3F* data);
	void setVertexNormal(Vector3F* data);
	
	void findCorner(int vi);
	void findEdge(int vi);
	void findInterior(int vi);

	void verbose() const;

	VertexAdjacency * m_vertexAdjacency;
	Vector3F* _positions;
	Vector3F* _normals;
	
	int m_patchVertices[4];
	
	AccCorner m_corners[4];
	AccEdge m_edges[4];
	AccInterior m_interiors[4];

private:
	void findFringeCornerNeighbors(int c, AccCorner & topo);
	char findSharedNeighbor(int a, int b, int c, int & dst);
};

