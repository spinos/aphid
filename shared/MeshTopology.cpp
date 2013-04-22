/*
 *  MeshTopology.cpp
 *  fit
 *
 *  Created by jian zhang on 4/22/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MeshTopology.h"
#include <BaseMesh.h>
#include <Facet.h>
#include <Edge.h>
#include <VertexAdjacency.h>

MeshTopology::MeshTopology() {}
MeshTopology::~MeshTopology() {}

char MeshTopology::buildTopology(BaseMesh * mesh)
{
	const unsigned nv = mesh->getNumVertices();
	
	m_topology = new VertexAdjacency[nv];
	
	for(unsigned i = 0; i < nv; i++) {
		VertexAdjacency & v = m_topology[i];
		v.setIndex(i);
		v.m_v = &(mesh->getVertices()[i]);
	}
	
	const unsigned nf = mesh->getNumFaces();
	unsigned a, b, c;
	
	for(unsigned i = 0; i < nf; i++) {
		a = mesh->getIndices()[i * 3];
		b = mesh->getIndices()[i * 3 + 1];
		c = mesh->getIndices()[i * 3 + 2];
		Facet * f = new Facet(&m_topology[a], &m_topology[b], &m_topology[c]);
		f->setIndex(i);
		for(unsigned j = 0; j < 3; j++) {
			Edge * e = f->edge(j);
			m_topology[e->v0()->getIndex()].addEdge(e);
			m_topology[e->v1()->getIndex()].addEdge(e);
		}
	}
	
	for(unsigned i = 0; i < nv; i++) {
		m_topology[i].findNeighbors();
		m_topology[i].computeWeights();
		m_topology[i].computeDifferentialCoordinate();
	}
	return 1;
}

VertexAdjacency * MeshTopology::getTopology() const
{
	return m_topology;
}
