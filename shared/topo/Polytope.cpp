/*
 *  Polytope.cpp
 *  convexHull
 *
 *  Created by jian zhang on 9/10/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "Polytope.h"
#include <topo/Vertex.h>
#include <topo/Facet.h>

namespace aphid {

Polytope::Polytope() 
{}

Polytope::~Polytope() 
{}

void Polytope::destroy()
{
	m_vertices.clear();
	m_faces.clear();
}

int Polytope::getNumVertex() const
{
	return m_vertices.size();
}
	
int Polytope::getNumFace() const
{
	return m_faces.size();
}

void Polytope::addVertex(Vertex * p)
{
	p->setIndex(getNumVertex());
	m_vertices.push_back(p);
}

void Polytope::addFacet(Facet * f)
{
	f->setIndex(getNumFace());
	m_faces.push_back(f);
#ifndef NDEBUG
	printf("add face %d\n", f->getIndex());
#endif
}

void Polytope::removeFaces()
{
	//printf("remove face\n");
	//printf("b4\n");
	std::vector<Facet *>::iterator it;
	std::vector<Facet *>::iterator rest;
	//for(it = m_faces.begin(); it < m_faces.end(); it++ )
	//	printf("%d ", (*it)->getIndex());

	int i = 0;
	for(it = m_faces.begin(); it < m_faces.end();)
	{ 
		if((*it)->getIndex() < 0)
		{
			for(rest = m_faces.begin() + i; rest < m_faces.end(); rest++)
			{
				(*rest)->setIndex((*rest)->getIndex() - 1);
			}
			
			(*it)->clear();
			m_faces.erase(m_faces.begin() + i);
		}
		else {
		    it++;
		    i++;
		}
	}
	
	//printf("\naft\n");
	//for(it = m_faces.begin(); it < m_faces.end(); it++ )
	//	printf("%d ", (*it)->getIndex());

	//printf("\n");
	
}

const Facet & Polytope::getFacet(int idx) const
{
	return *m_faces[idx];
}

const Vertex & Polytope::getVertex(int idx) const
{
	return *m_vertices[idx];
}

Vertex * Polytope::vertex(int idx)
{
	return m_vertices[idx];
}

std::vector<Facet *> & Polytope::faces()
{ return m_faces; }

}
