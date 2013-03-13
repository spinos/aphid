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
class VertexAdjacency : public Vertex {
public:
	VertexAdjacency();
	virtual ~VertexAdjacency();
	
	void addEdge(Edge * e);
	
	char findOneRingNeighbors();
	void computeWeights();
	void computeNormal();
	
	char findOppositeEdge(Edge & e, Edge & dest) const;
	char firstOutgoingEdge(Edge & e);
	char findIncomming(Edge & eout, Edge & ein);
	
	std::map<int,int> getNeighborOrder() const;
	
	float getDeltaCoordX() const;
	float getDeltaCoordY() const;
	float getDeltaCoordZ() const;

	void getNeighbor(const int & idx, int & vertexIdx, float & weight) const;
	void verbose() const;
private:
    void getVijs(const int & idx, Vector3F &vij, Vector3F &vij0, Vector3F &vij1) const;
	std::vector<Edge *> m_edges;
	std::vector<Vertex *> m_neighbors;
	std::vector<float> m_weights;
	std::map<int,int> m_idxInOrder;
	Vector3F m_mvcoord, m_normal;
};
