/*
 *  SimpleTopology.cpp
 *  fit
 *
 *  Created by jian zhang on 4/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "SimpleTopology.h"
#include <topo/Facet.h>
#include <topo/Edge.h>
#include <topo/VertexAdjacency.h>

namespace aphid {

SimpleTopology::SimpleTopology(sdb::PNPrefW * pool, int * tri, const int & numV, const int & numTri)
{
    m_pool = pool;
	m_nv = numV;
    m_adjacency.reset(new VertexAdjacency[m_nv]);
    
	int i;
    for(i = 0; i < m_nv; i++) {
		VertexAdjacency & v = m_adjacency[i];
		v.setIndex(i);
		v.m_v = m_pool[i].t1;
	}
	
	unsigned a, b, c;
	const int nf = numTri;
	for(i = 0; i < nf; i++) {
		a = tri[i * 3];
		b = tri[i * 3 + 1];
		c = tri[i * 3 + 2];
		Facet * f = new Facet(&m_adjacency[a], &m_adjacency[b], &m_adjacency[c]);
		f->setIndex(i);
		f->setPolygonIndex(i);
		for(unsigned j = 0; j < 3; j++) {
			Edge * e = f->edge(j);
			m_adjacency[e->v0()->getIndex()].addEdge(e);
			m_adjacency[e->v1()->getIndex()].addEdge(e);
		}
		m_faces.push_back(f);
	}
	
	for(i = 0; i < m_nv; i++) {
		m_adjacency[i].findNeighbors();
		m_adjacency[i].connectEdges();
	}
	
	for(i = 0; i < m_nv; i++)
		m_adjacency[i].computeWeights();
}

SimpleTopology::~SimpleTopology() 
{
	cleanup();
}

void SimpleTopology::calculateWeight()
{
	for(int i = 0; i < m_nv; i++) {
		m_adjacency[i].computeWeights();
		m_adjacency[i].computeDifferentialCoordinate();
	}
}

void SimpleTopology::calculateNormal()
{
	for(std::vector<Facet *>::iterator it = m_faces.begin(); it != m_faces.end(); ++it) {
		(*it)->update();
	}
	
	for(int i = 0; i < m_nv; i++) {
		*m_pool[i].t2 = m_adjacency[i].computeNormal();
	}
}

void SimpleTopology::calculateVertexNormal(const int & i)
{
    std::vector<unsigned> faceid;
    m_adjacency[i].getConnectedFacets(faceid);
    std::vector<unsigned>::iterator it = faceid.begin();
    for(; it != faceid.end(); ++it) m_faces[(*it)]->update();
    
    *m_pool[i].t2 = m_adjacency[i].computeNormal();
}

VertexAdjacency * SimpleTopology::getTopology() const
{
	return m_adjacency.get();
}

VertexAdjacency & SimpleTopology::getAdjacency(unsigned idx) const
{
	return m_adjacency[idx];
}

Facet * SimpleTopology::getFacet(unsigned idx) const
{
	return m_faces[idx];
}

Edge * SimpleTopology::getEdge(unsigned idx) const
{
	const unsigned facetIdx = idx / 3;
	const unsigned edgeInFacetIdx = idx - facetIdx * 3;
	return getFacet(facetIdx)->edge(edgeInFacetIdx);
}

Edge * SimpleTopology::findEdge(unsigned a, unsigned b) const
{
	VertexAdjacency & adj = getAdjacency(a);
	char found = 0;
	return adj.outgoingEdgeToVertex(b, found);
}

void SimpleTopology::cleanup()
{
	m_adjacency.reset();
	for(std::vector<Facet *>::iterator it = m_faces.begin(); it != m_faces.end(); ++it) {
		delete *it;
	}
	m_faces.clear();
}

void SimpleTopology::update(const int & nv)
{
	for(std::vector<Facet *>::iterator it = m_faces.begin(); it != m_faces.end(); ++it) {
		(*it)->update();
	}
	
	for(int i = 0; i < nv; i++) {
		m_adjacency[i].computeWeights();
		*m_pool[i].t2 = m_adjacency[i].computeNormal();
	}
}

void SimpleTopology::getDifferentialCoord(const int & vertexId, Vector3F & dst)
{
	VertexAdjacency & adj = m_adjacency[vertexId];
	adj.computeDifferentialCoordinate();
	dst = adj.getDifferentialCoordinate();
}

}
//:~
