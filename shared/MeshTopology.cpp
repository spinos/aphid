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
	m_mesh = mesh;
	const unsigned nv = mesh->getNumVertices();
	
	m_adjacency = new VertexAdjacency[nv];
	
	for(unsigned i = 0; i < nv; i++) {
		VertexAdjacency & v = m_adjacency[i];
		v.setIndex(i);
		v.m_v = &(mesh->getVertices()[i]);
	}
	
	unsigned a, b, c;
	
	unsigned * polyC = mesh->m_polygonCounts;
	unsigned polyTri = *polyC - 2;
	unsigned curTri = 0;
	unsigned curPoly = 0;
	
	const unsigned nf = mesh->getNumTriangles();
	for(unsigned i = 0; i < nf; i++) {
		a = mesh->getIndices()[i * 3];
		b = mesh->getIndices()[i * 3 + 1];
		c = mesh->getIndices()[i * 3 + 2];
		Facet * f = new Facet(&m_adjacency[a], &m_adjacency[b], &m_adjacency[c]);
		f->setIndex(i);
		f->setPolygonIndex(curPoly);
		for(unsigned j = 0; j < 3; j++) {
			Edge * e = f->edge(j);
			m_adjacency[e->v0()->getIndex()].addEdge(e);
			m_adjacency[e->v1()->getIndex()].addEdge(e);
		}
		m_faces.push_back(f);
		
		curTri++;
		if(curTri == polyTri) {
			polyC++;
			polyTri = *polyC - 2;
			curPoly++;
			curTri = 0;
		}
	}
	
	for(unsigned i = 0; i < nv; i++) {
		m_adjacency[i].findNeighbors();
		m_adjacency[i].connectEdges();
	}
	
	calculateWeight();
	return 1;
}

void MeshTopology::calculateWeight()
{
	const unsigned nv = m_mesh->getNumVertices();
	
	for(unsigned i = 0; i < nv; i++) {
		m_adjacency[i].computeWeights();
		m_adjacency[i].computeDifferentialCoordinate();
	}
}

void MeshTopology::calculateNormal()
{
	for(std::vector<Facet *>::iterator it = m_faces.begin(); it != m_faces.end(); ++it) {
		(*it)->update();
	}
	
	const unsigned nv = m_mesh->getNumVertices();
	for(unsigned i = 0; i < nv; i++) {
		m_mesh->normals()[i] = m_adjacency[i].computeNormal();
	}
}

VertexAdjacency * MeshTopology::getTopology() const
{
	return m_adjacency;
}

VertexAdjacency & MeshTopology::getAdjacency(unsigned idx) const
{
	return m_adjacency[idx];
}

Facet * MeshTopology::getFacet(unsigned idx) const
{
	return m_faces[idx];
}

Edge * MeshTopology::getEdge(unsigned idx) const
{
	const unsigned facetIdx = idx / 3;
	const unsigned edgeInFacetIdx = idx - facetIdx * 3;
	return getFacet(facetIdx)->edge(edgeInFacetIdx);
}

Edge * MeshTopology::findEdge(unsigned a, unsigned b) const
{
	VertexAdjacency & adj = getAdjacency(a);
	char found = 0;
	return adj.outgoingEdgeToVertex(b, found);
}

Edge * MeshTopology::parallelEdge(Edge * src) const
{
	Facet * face = (Facet *)src->getFace();
	unsigned polyIdx = face->getPolygonIndex();
	unsigned * pv = m_mesh->quadIndices();
	pv += polyIdx * 4;
	
	unsigned v0 = src->v0()->getIndex();
	unsigned v1 = src->v1()->getIndex();
	unsigned a, b;
	char found = parallelEdgeInQuad(pv, v0, v1, a, b);
	
	if(!found || a == b) return 0;
	
	return findEdge(a, b);
}

char MeshTopology::parallelEdgeInQuad(unsigned *indices, unsigned v0, unsigned v1, unsigned & a, unsigned & b) const
{
	int i, j;
	for(i = 0; i < 4; i++) {
		j = (i + 1)%4;
		if(indices[i] == v0 && indices[j] == v1) {
			a = indices[(j+1)%4];
			b = indices[(j+2)%4];
			return 1;
		}
	}
	
	for(i = 3; i >= 0; i--) {
		j = i - 1;
		if(j < 0) j = 3;
		if(indices[i] == v0 && indices[j] == v1) {
			a = indices[(j-1)%4];
			b = indices[(j-2)%4];
			return 1;
		}
	}
	
	return 0;
}

void MeshTopology::cleanup()
{
	if(m_adjacency) delete[] m_adjacency;
	m_adjacency = 0;
	for(std::vector<Facet *>::iterator it = m_faces.begin(); it != m_faces.end(); ++it) {
		delete *it;
	}
	m_faces.clear();
}

unsigned MeshTopology::growAroundQuad(unsigned idx, std::vector<unsigned> & dst) const
{
	const unsigned presize = dst.size();
	unsigned * pv = m_mesh->quadIndices();
	pv += idx * 4;
	
	unsigned iv;
	for(unsigned i = 0; i < 4; i++) {
		iv = pv[i];
		VertexAdjacency & adj = getAdjacency(iv);
		adj.getConnectedPolygons(dst);
	}
	
	return dst.size() - presize;
}

void MeshTopology::calculateSmoothedNormal(Vector3F * dst)
{
	Vector3F * nor = m_mesh->normals();
	Vector3F c;
	const unsigned nv = m_mesh->getNumVertices();
	int neighborIdx;
	for(unsigned i = 0; i < nv; i++) {
		c.setZero();
		VertexAdjacency & adj = getAdjacency(i);
		VertexAdjacency::VertexNeighbor *neighbor;
		for(neighbor = adj.firstNeighbor(); !adj.isLastNeighbor(); neighbor = adj.nextNeighbor()) {
			neighborIdx = neighbor->v->getIndex();
			c += nor[neighborIdx] * neighbor->weight;
		}
		dst[i] = c.normal();
	}
}
//:~
