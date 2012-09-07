/*
 *  ConflictGraph.cpp
 *  convexHull
 *
 *  Created by jian zhang on 9/6/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "ConflictGraph.h"

ConflictGraph::ConflictGraph(char faceOriented) 
{
	m_head = 0;
	m_faceOriented = faceOriented;
}

ConflictGraph::~ConflictGraph() {}

void ConflictGraph::clear()
{
	delete m_head;
	m_head = 0;
}

void ConflictGraph::add(GraphArch * arch)
{
	if(m_faceOriented)
	{
		if(m_head) { m_head->previousFace = arch; }
		arch->nextFace = m_head;
		m_head = arch;
	} 
	else
	{
		if(m_head) { m_head->previousVertex = arch; }
		arch->nextVertex = m_head;
		m_head = arch;
	}
}

void ConflictGraph::getFaces(std::vector<Facet *>&faces) const
{
	GraphArch *arch = m_head;
	while(arch)
	{
		if(arch->face->getIndex() > -1) faces.push_back(arch->face);
		arch = arch->nextVertex;
	}
}

void ConflictGraph::getVertices(std::vector<Vertex *>&vertices) const
{
	GraphArch *arch = m_head;
	while(arch)
	{
		vertices.push_back(arch->vertex);
		arch = arch->nextFace;
	}
}
