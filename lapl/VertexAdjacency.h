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
	void computeDiscreteForms();

	std::map<int,int> getNeighborOrder() const;
	
	float getDeltaCoordX() const;
	float getDeltaCoordY() const;
	float getDeltaCoordZ() const;
	unsigned getNumNeighbors() const;
	void getNeighbor(const int & idx, int & vertexIdx, float & weight) const;
	Matrix33F getTangentFrame() const;
	void verbose() const;
private:
	char findOppositeEdge(Edge & e, Edge & dest) const;
	char firstOutgoingEdge(Edge & e) const;
	char findIncomming(Edge & eout, Edge & ein) const;
	void addNeighbor(Edge &e);
    void getVijs(const int & idx, Vector3F &vij, Vector3F &vij0, Vector3F &vij1) const;
	
	std::vector<Edge *> m_edges;
	std::vector<VertexNeighbor *> m_neighbors;
	std::vector<float> m_g1;
	std::vector<float> m_g2;
	std::vector<float> m_g3;
	std::vector<float> m_L;
	std::vector<float> m_O;
	std::vector<Vector3F> m_x_bar;
	std::map<int,int> m_idxInOrder;
	Matrix33F m_tangentFrame;
	Vector3F m_mvcoord;
	Vector3F m_normal;
	
};
