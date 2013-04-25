/*
 *  MeshTopology.h
 *  fit
 *
 *  Created by jian zhang on 4/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <vector>
class VertexAdjacency;
class Facet;
class BaseMesh;

class MeshTopology {
public:
	MeshTopology();
	virtual ~MeshTopology();
	
	char buildTopology(BaseMesh * mesh);
	void calculateNormal(BaseMesh * mesh);

	VertexAdjacency * getTopology() const;
private:
	void cleanup();
	VertexAdjacency * m_adjacency;
	std::vector<Facet *> m_faces;
};