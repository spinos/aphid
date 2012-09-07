/*
 *  Facet.cpp
 *  convexHull
 *
 *  Created by jian zhang on 9/5/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "Facet.h"

Facet::Facet() {}

Facet::Facet(Vertex *a, Vertex *b, Vertex *c, Vertex *d)
{
	m_vertices[0] = a;
	m_vertices[1] = b;
	m_vertices[2] = c;
	
	Vector3F e0 = *b - *a;
	Vector3F e1 = *c - *a;
	m_normal = e0.cross(e1);
	m_normal.normalize();
	
	Vector3F e2 = *d - *a;
	if(e2.dot(m_normal) > 0.f)
		m_normal.reverse();
		
	createEdges();
}

Facet::~Facet()
{
	
}

void Facet::createEdges()
{
	m_edges[0] = new Edge(m_vertices[0], m_vertices[1], (char*)this);
	m_edges[1] = new Edge(m_vertices[1], m_vertices[2], (char*)this);
	m_edges[2] = new Edge(m_vertices[2], m_vertices[0], (char*)this);
}

void Facet::connectTo(Facet *another, Vertex *a, Vertex *b)
{

	Edge *inner = matchedEdge(a, b);
    Edge *outter = another->matchedEdge(a, b);
	inner->setTwin(outter);
	outter->setTwin(inner);
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

Vertex * Facet::vertex(int idx)
{
	return m_vertices[idx];
}

Vertex * Facet::thirdVertex(Vertex *a, Vertex *b)
{
	if(!m_vertices[0]->equals(*a) && !m_vertices[0]->equals(*b)) return m_vertices[0];
	if(!m_vertices[1]->equals(*a) && !m_vertices[1]->equals(*b)) return m_vertices[1];
	return m_vertices[2];
}

Vertex Facet::getVertex(int idx) const
{
	return *m_vertices[idx];
}

Vector3F Facet::getCentroid() const
{
	return (*m_vertices[0] * 0.333f + *m_vertices[1] * 0.333f + *m_vertices[2] * 0.333f);
}

Vector3F Facet::getNormal() const
{
	return m_normal;
}

char Facet::isVertexAbove(const Vertex & v) const
{
	const Vector3F d = v - getCentroid();
	return d.dot(m_normal) > 0.f;
}

void Facet::getEdgeOnHorizon(std::vector<Edge *> & horizons) const
{
	for (int i=0; i<3; i++) 
	{
         Edge *opposite = m_edges[i]->getTwin();
		 if(!opposite) continue;
		 Facet * f = (Facet *)(opposite->getFace());
		 
		 if(!f) continue;
		 
		 if(!f->isMarked()) 
		{
			horizons.push_back(opposite);
		}
	}
}

