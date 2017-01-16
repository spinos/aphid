/*
 *  SimpleTopology.h
 *  fit
 *
 *  Created by jian zhang on 4/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <sdb/Types.h>
#include <vector>
#include <boost/scoped_array.hpp>

namespace aphid {

class VertexAdjacency;
class Facet;
class Edge;

class SimpleTopology {
	int m_nv;
public:
	SimpleTopology(sdb::PNPrefW * pool, int * tri, const int & numV, const int & numTri);
    virtual ~SimpleTopology();
	
	void cleanup();
	
	void update(const int & nv);
	void getDifferentialCoord(const int & vertexId, Vector3F & dst);
	void calculateWeight();
	void calculateNormal();
	void calculateVertexNormal(const int & i);

	VertexAdjacency * getTopology() const;
	VertexAdjacency & getAdjacency(unsigned idx) const;
	Facet * getFacet(unsigned idx) const;
	Edge * getEdge(unsigned idx) const;
	Edge * findEdge(unsigned a, unsigned b) const;
	
private:
	boost::scoped_array<VertexAdjacency> m_adjacency;
	std::vector<Facet *> m_faces;
	sdb::PNPrefW * m_pool;
};

}