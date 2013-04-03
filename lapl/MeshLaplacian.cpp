#include "MeshLaplacian.h"
#include "Facet.h"
#include "VertexAdjacency.h"

MeshLaplacian::MeshLaplacian() {}

MeshLaplacian::MeshLaplacian(const char * filename) : TriangleMesh(filename), m_adjacency(NULL)
{
	buildTopology();
}

MeshLaplacian::~MeshLaplacian() 
{
	if(m_adjacency) delete[] m_adjacency;
}

char MeshLaplacian::buildTopology()
{
	const unsigned nv = getNumVertices();
	
	m_adjacency = new VertexAdjacency[nv];
	
	for(unsigned i = 0; i < nv; i++) {
		VertexAdjacency & v = m_adjacency[i];
		v.setIndex(i);
		v.m_v = &_vertices[i];
	}
	
	const unsigned nf = getNumFaces();
	unsigned a, b, c;
	
	for(unsigned i = 0; i < nf; i++) {
		a = _indices[i * 3];
		b = _indices[i * 3 + 1];
		c = _indices[i * 3 + 2];
		Facet * f = new Facet(&m_adjacency[a], &m_adjacency[b], &m_adjacency[c]);
		f->setIndex(i);
		for(unsigned j = 0; j < 3; j++) {
			Edge * e = f->edge(j);
			m_adjacency[e->v0()->getIndex()].addEdge(e);
			m_adjacency[e->v1()->getIndex()].addEdge(e);
		}
		m_faces.push_back(f);
	}
	
	for(unsigned i = 0; i < nv; i++) {
		m_adjacency[i].findNeighbors();
		m_adjacency[i].computeWeights();
		m_adjacency[i].computeDifferentialCoordinate();
		//m_adjacency[i].verbose();
	}
	return 1;
}

VertexAdjacency * MeshLaplacian::connectivity()
{
	return m_adjacency;
}

Matrix33F MeshLaplacian::getTangentFrame(const unsigned &idx) const
{
    return m_adjacency[idx].getTangentFrame();
}

Facet * MeshLaplacian::getFace(const unsigned &idx) const
{
	if(idx >= getNumFaces()) return 0;
	return m_faces[idx];
}

Vertex * MeshLaplacian::getVertex(const unsigned &idx) const
{
    if(idx >= getNumVertices()) return 0;
    return &m_adjacency[idx];
}
