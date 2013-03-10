/*
 *  VertexAdjacency.h
 *  lapl
 *
 *  Created by jian zhang on 3/9/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <vector>
#include "Vertex.h"
class Edge;
class VertexAdjacency : public Vertex {
public:
	VertexAdjacency();
	virtual ~VertexAdjacency();
	
	void addEdge(Edge * e);
	
	char checkOneRing() const;

	char findOppositeEdge(Edge & e, Edge & dest) const;
	char findOneRingNeighbors();
	char firstOutgoingEdge(Edge & e);
	char findIncomming(Edge & eout, Edge & ein);

	void verbose() const;
private:
	std::vector<Edge *> m_edges;
	std::vector<Vertex *> m_neighbors;
};
