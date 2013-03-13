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
#include <map>
#include "Vertex.h"
class Edge;
class Facet;
class VertexAdjacency : public Vertex {
public:
	struct VertexNeighbor {
		Vertex *v;
		Edge *e;
		Facet *f;
		float weight;
	};
	
	VertexAdjacency();
	virtual ~VertexAdjacency();
	
	void addEdge(Edge * e);
	
	char findOneRingNeighbors();
	void computeWeights();
	void computeTangentFrame();

	std::map<int,int> getNeighborOrder() const;
	
	float getDeltaCoordX() const;
	float getDeltaCoordY() const;
	float getDeltaCoordZ() const;

	void getNeighbor(const int & idx, int & vertexIdx, float & weight) const;
	void verbose() const;
private:
	char findOppositeEdge(Edge & e, Edge & dest) const;
	char firstOutgoingEdge(Edge & e) const;
	char findIncomming(Edge & eout, Edge & ein) const;
	void addNeighbor(Edge &e);
    void getVijs(const int & idx, Vector3F &vij, Vector3F &vij0, Vector3F &vij1) const;
	
	std::vector<Edge *> m_edges;
	std::vector<VertexNeighbor *> m_neighbors;
	std::map<int,int> m_idxInOrder;
	Vector3F m_mvcoord, m_tangent, m_binormal, m_normal;
};
