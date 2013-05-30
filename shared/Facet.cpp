/*
 *  Facet.cpp
 *  convexHull
 *
 *  Created by jian zhang on 9/5/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <cmath>
#include "Facet.h"

Facet::Facet() {}

Facet::Facet(Vertex *a, Vertex *b, Vertex *c)
{
	m_vertices[0] = a;
	m_vertices[1] = b;
	m_vertices[2] = c;
	
	Vector3F e0 = *b->m_v - *a->m_v; e0.normalize();
	Vector3F e1 = *c->m_v - *a->m_v; e1.normalize();
	m_normal = e0.cross(e1);
	m_normal.normalize();
		
	createEdges();
	
	m_area = Facet::cumputeArea(a->m_v, b->m_v, c->m_v);
	if(m_area < 10e-5) printf("WARNING: near zero face area!\n");
}

Facet::Facet(Vertex *a, Vertex *b, Vertex *c, Vector3F *d)
{
	m_vertices[0] = a;
	m_vertices[1] = b;
	m_vertices[2] = c;
	
	Vector3F e0 = *b->m_v - *a->m_v; e0.normalize();
	Vector3F e1 = *c->m_v - *a->m_v; e1.normalize();
	m_normal = e0.cross(e1);
	m_normal.normalize();
	
	Vector3F e2 = *d - *a->m_v;
	if(e2.dot(m_normal) > 0.f)
		m_normal.reverse();
		
	createEdges();
	
	m_area = Facet::cumputeArea(a->m_v, b->m_v, c->m_v);
	if(m_area < 10e-5) printf("WARNING: near zero face area!\n");
}

Facet::~Facet()
{
	clear();
}

void Facet::clear()
{
	delete m_edges[0];
	delete m_edges[1];
	delete m_edges[2];
}

void Facet::createEdges()
{
	m_edges[0] = new Edge(m_vertices[0], m_vertices[1], (char*)this);
	m_edges[1] = new Edge(m_vertices[1], m_vertices[2], (char*)this);
	m_edges[2] = new Edge(m_vertices[2], m_vertices[0], (char*)this);
}

char Facet::connectTo(Facet *another, Vertex *a, Vertex *b)
{
	Edge *inner = matchedEdge(a, b);
	if(!inner) return 0;
    Edge *outter = another->matchedEdge(a, b);
	if(!outter) return 0;
	inner->setTwin(outter);
	outter->setTwin(inner);
	return 1;
}

Edge * Facet::matchedEdge(Vertex * a, Vertex * b)
{
	for (int i=0; i<3; i++) 
	{
         if (m_edges[i]->matches(a, b)) 
		 {
			return m_edges[i];
		}
	}
	return 0;
}

Edge * Facet::edge(int idx)
{
	return m_edges[idx];
}

Vertex * Facet::vertex(int idx)
{
	return m_vertices[idx];
}

Vertex * Facet::vertexAfter(int idx)
{
	if(idx == 2) return m_vertices[0];
	return m_vertices[idx + 1];
}

Vertex * Facet::vertexBefore(int idx)
{
	if(idx == 0) return m_vertices[2];
	return m_vertices[idx - 1];
}

Vertex * Facet::thirdVertex(Vertex *a, Vertex *b)
{
	if(!m_vertices[0]->getIndex() == a->getIndex() && !m_vertices[0]->getIndex() == b->getIndex()) return m_vertices[0];
	if(!m_vertices[1]->getIndex() == a->getIndex() && !m_vertices[1]->getIndex() == b->getIndex()) return m_vertices[1];
	return m_vertices[2];
}

Vertex Facet::getVertex(int idx) const
{
	return *m_vertices[idx];
}

Vector3F Facet::getCentroid() const
{
	return (*m_vertices[0]->m_v * 0.333f + *m_vertices[1]->m_v * 0.333f + *m_vertices[2]->m_v * 0.333f);
}

Vector3F Facet::getNormal() const
{
	return m_normal;
}

float Facet::getArea() const
{
	return m_area;
}

char Facet::isVertexAbove(const Vertex & v) const
{
	Vector3F dv = *v.m_v - getCentroid();
	dv.normalize();
	return dv.dot(m_normal) > 0.0f;
}

char Facet::getEdgeOnHorizon(std::vector<Edge *> & horizons) const
{
	for (int i=0; i<3; i++) 
	{
         Edge *opposite = m_edges[i]->getTwin();
		 if(!opposite) 
		 {
			printf("edge not connected\n");
			return 0;
		}
		 Facet * f = (Facet *)(opposite->getFace());
		 
		 if(!f){
			printf("edge has no face\n");
			return 0;
		}
		
		if(f->getIndex() < 0)
		{
			printf("face %d is removed\n", f->getIndex());
			return 0;
		}
		 
		 if(!f->isMarked()) 
		{
			horizons.push_back(opposite);
		}
	}
	
	return 1;
}

char Facet::isClosed() const
{
	for (int i=0; i<3; i++) 
	{
         Edge *opposite = m_edges[i]->getTwin();
		 if(!opposite) 
		 {
			printf("edge not connected\n");
			return 0;
		}
		 Facet * f = (Facet *)(opposite->getFace());
		 
		 if(!f){
			printf("edge has no face\n");
			return 0;
		}
		
		if(f->getIndex() < 0)
		{
			printf("face %d is removed\n", f->getIndex());
			return 0;
		}
		 
	}
	
	return 1;
}

void Facet::update()
{
	Vector3F e0 = *m_vertices[1]->m_v - *m_vertices[0]->m_v; e0.normalize();
	Vector3F e1 = *m_vertices[2]->m_v - *m_vertices[0]->m_v; e1.normalize();
	m_normal = e0.cross(e1);
	m_normal.normalize();
	
	m_area = Facet::cumputeArea(m_vertices[0]->m_v, m_vertices[1]->m_v, m_vertices[2]->m_v);
	if(m_area < 10e-5) printf("WARNING: near zero face area!\n");
}

void Facet::setPolygonIndex(unsigned idx)
{
	m_polyIndex = idx;
}

unsigned Facet::getPolygonIndex() const
{
	return m_polyIndex;
}

float Facet::cumputeArea(Vector3F *a, Vector3F *b, Vector3F *c)
{
    Vector3F d[3]; float l[3];

	d[0] = *b - *a;
	l[0] = d[0].length();
	
	d[1] = *c - *b;
	l[1] = d[1].length();
	
	d[2] = *a - *c;
	l[2] = d[2].length();
	
	return (0.5*sqrt(l[0]*l[0]*l[2]*l[2] - (d[0].dot(d[2]))*(d[0].dot(d[2]))));
}
