/*
 *  Polytode.cpp
 *  convexHull
 *
 *  Created by jian zhang on 9/10/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "Polytode.h"

Polytode::Polytode() {}
Polytode::~Polytode() {}

void Polytode::destroy()
{
	m_vertices.clear();
	m_faces.clear();
}

int Polytode::getNumVertex() const
{
	return m_vertices.size();
}
	
int Polytode::getNumFace() const
{
	return m_faces.size();
}

void Polytode::addVertex(Vertex *p)
{
	p->setIndex(getNumVertex());
	m_vertices.push_back(p);
}

void Polytode::addFacet(Facet *f)
{
	f->setIndex(getNumFace());
	m_faces.push_back(f);
#ifndef NDEBUG
	printf("add face %d\n", f->getIndex());
#endif
}

void Polytode::removeFaces()
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

Facet Polytode::getFacet(int idx) const
{
	return *m_faces[idx];
}

Vertex Polytode::getVertex(int idx) const
{
	return *m_vertices[idx];
}

Vertex *Polytode::vertex(int idx)
{
	return m_vertices[idx];
}
