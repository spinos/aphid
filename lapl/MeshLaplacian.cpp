#include "MeshLaplacian.h"
#include "Facet.h"
#include "VertexAdjacency.h"
MeshLaplacian::MeshLaplacian() {}

MeshLaplacian::MeshLaplacian(const char * filename) : TriangleMesh(filename), m_adjacency(NULL)
{
	const unsigned nv = getNumVertices();
	
	m_adjacency = new VertexAdjacency[nv];
	
	std::vector<Vertex *> verts;
	for(unsigned i = 0; i < nv; i++) {
		Vertex * v = new Vertex;
		v->setIndex(i);
		verts.push_back(v);
	}
	
	const unsigned nf = getNumFaces();
	unsigned a, b, c;
	std::vector<Facet *> faces;
	for(unsigned i = 0; i < nf; i++) {
		a = _indices[i * 3];
		b = _indices[i * 3 + 1];
		c = _indices[i * 3 + 2];
		Facet * f = new Facet(verts[a], verts[b], verts[c]);
		for(unsigned j = 0; j < 3; j++) {
			Edge * e = f->edge(j);
			m_adjacency[e->v0()->getIndex()].addEdge(e, e->v0()->getIndex());
			m_adjacency[e->v1()->getIndex()].addEdge(e, e->v1()->getIndex());
		}
		faces.push_back(f);
	}
	
	for(unsigned i = 0; i < nv; i++) {
		printf("v %d ", i);
		m_adjacency[i].verbose();
	}
	/*
	std::vector<Facet *>::iterator it;
	for(it = faces.begin(); it < faces.end(); it++) {
		Facet * f = (*it);
		for(unsigned i = 0; i < 3; i++) {
			Vertex * v = f->vertex(i);
			Vertex * v0 = f->vertexBefore(i);
			Vertex * v1 = f->vertexAfter(i);
		}
	}*/
}

MeshLaplacian::~MeshLaplacian() 
{
	if(m_adjacency) delete[] m_adjacency;
}
    
