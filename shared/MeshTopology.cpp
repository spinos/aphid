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

MeshTopology::MeshTopology() 
{
	m_adjacency = 0;
}

MeshTopology::~MeshTopology() 
{
	cleanup();
}

char MeshTopology::buildTopology(BaseMesh * mesh)
{
	cleanup();
	const unsigned nv = mesh->getNumVertices();
	
	m_adjacency = new VertexAdjacency[nv];
	
	for(unsigned i = 0; i < nv; i++) {
		VertexAdjacency & v = m_adjacency[i];
		v.setIndex(i);
		v.m_v = &(mesh->getVertices()[i]);
	}
	
	const unsigned nf = mesh->getNumFaces();
	unsigned a, b, c;
	
	for(unsigned i = 0; i < nf; i++) {
		a = mesh->getIndices()[i * 3];
		b = mesh->getIndices()[i * 3 + 1];
		c = mesh->getIndices()[i * 3 + 2];
		Facet * f = new Facet(&m_adjacency[a], &m_adjacency[b], &m_adjacency[c]);
		f->setIndex(i);
		for(unsigned j = 0; j < 3; j++) {
			Edge * e = f->edge(j);
			e->setReal(mesh->m_realEdges[i * 3 + j]);
			m_adjacency[e->v0()->getIndex()].addEdge(e);
			m_adjacency[e->v1()->getIndex()].addEdge(e);
		}
		m_faces.push_back(f);
	}
	
	for(unsigned i = 0; i < nv; i++) {
		m_adjacency[i].findNeighbors();
		m_adjacency[i].computeWeights();
		m_adjacency[i].computeDifferentialCoordinate();
	}
	return 1;
}

void MeshTopology::calculateNormal(BaseMesh * mesh)
{
	for(std::vector<Facet *>::iterator it = m_faces.begin(); it != m_faces.end(); ++it) {
		(*it)->update();
	}
	
	const unsigned nv = mesh->getNumVertices();
	for(unsigned i = 0; i < nv; i++) {
		mesh->normals()[i] = m_adjacency[i].computeNormal();
	}
}

VertexAdjacency * MeshTopology::getTopology() const
{
	return m_adjacency;
}

void MeshTopology::cleanup()
{
	if(m_adjacency) delete[] m_adjacency;
	for(std::vector<Facet *>::iterator it = m_faces.begin(); it != m_faces.end(); ++it) {
		delete *it;
	}
	m_faces.clear();
}
//:~
