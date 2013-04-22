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
class VertexAdjacency;
class BaseMesh;

class MeshTopology {
public:
	MeshTopology();
	virtual ~MeshTopology();
	
	char buildTopology(BaseMesh * mesh);

	VertexAdjacency * getTopology() const;
	
	VertexAdjacency * m_topology;
private:
};