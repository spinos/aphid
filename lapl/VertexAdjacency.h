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
#include "Matrix33F.h"
class Edge;
class Facet;
class VertexAdjacency : public Vertex {
public:
	struct VertexNeighbor {
		Vertex *v;
		Facet *f;
		float weight;
	};
	
	VertexAdjacency();
	virtual ~VertexAdjacency();
	
	void addEdge(Edge * e);
	
	char isOpen() const;
	void findNeighbors();
	void computeWeights();
	void computeDifferentialCoordinate();
	void computeTangentFrame();
	Vector3F computeNormal();

	unsigned getNumNeighbors() const;
	Matrix33F getTangentFrame() const;
	Vector3F getDifferentialCoordinate() const;
	
	VertexNeighbor * firstNeighbor();
	VertexNeighbor * nextNeighbor();
	char isLastNeighbor();
	
	VertexNeighbor * firstNeighborOrderedByVertexIdx();
	VertexNeighbor * nextNeighborOrderedByVertexIdx();
	char isLastNeighborOrderedByVertexIdx();
	
	bool isConnectedTo(unsigned idx);
	unsigned nextRealEdgeNeighbor(unsigned idx);
	unsigned nextBoundaryNeighbor(unsigned idx);
	char findEdge(int i, int j, Edge & dest) const;
	
	void verbose() const;
private:
	char findOneRingNeighbors();
	void findOpenNeighbors();
	char findOppositeEdge(int i, int j, Edge & dest) const;
	char findOppositeEdge(Edge & e, Edge & dest) const;
	char firstOutgoingEdge(Edge & e) const;
	char firstOutgoingEdgeOnBoundary(Edge & e) const;
	char findIncomming(Edge & eout, Edge & ein) const;
	void addNeighbor(Edge *e, char isOutgoing = 1);
    void getVijs(int idx, Vector3F &vij, Vector3F &vij0, Vector3F &vij1) const;
	
	std::vector<Edge *> m_edges;
	std::vector<VertexNeighbor *> m_neighbors;
	std::map<int,int> m_idxInOrder;
	Matrix33F m_tangentFrame;
	Vector3F m_normal;
	Vector3F m_differential;
	std::vector<VertexNeighbor *>::iterator m_neighborIt;
	std::map<int,int>::iterator m_orderedNeighborIt;
};
