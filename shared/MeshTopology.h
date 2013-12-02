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
class Edge;
class BaseMesh;

class MeshTopology {
public:
	MeshTopology();
	virtual ~MeshTopology();
	
	void cleanup();
	
	char buildTopology(BaseMesh * mesh);
	void calculateWeight(const unsigned & nv);
	void calculateNormal(BaseMesh * mesh);
	void calculateConcaveShell(BaseMesh * mesh);

	VertexAdjacency * getTopology() const;
	VertexAdjacency & getAdjacency(unsigned idx) const;
	Facet * getFacet(unsigned idx) const;
	Edge * getEdge(unsigned idx) const;
	Edge * findEdge(unsigned a, unsigned b) const;
	Edge * parallelEdge(Edge * src) const;
	unsigned growAroundQuad(unsigned idx, std::vector<unsigned> & dst) const;
private:
	char parallelEdgeInQuad(unsigned *indices, unsigned v0, unsigned v1, unsigned & a, unsigned & b) const;
	VertexAdjacency * m_adjacency;
	std::vector<Facet *> m_faces;
	BaseMesh * m_mesh;
};