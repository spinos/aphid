/*
 *  SelectionArray.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "SelectionArray.h"
#include "Primitive.h"
#include "Vertex.h"
#include "Facet.h"
#include "MeshLaplacian.h"

SelectionArray::SelectionArray() {}
SelectionArray::~SelectionArray() {}

void SelectionArray::reset() 
{
	m_prims.clear();
	m_vertices.clear();
}

void SelectionArray::add(Primitive * prim)
{
	std::vector<Primitive *>::iterator it;
	for(it = m_prims.begin(); it != m_prims.end(); ++it) {
		if((*it) == prim)
			return;
	}
	m_prims.push_back(prim);
	
	MeshLaplacian * mesh = (MeshLaplacian *)prim->getGeometry();
	unsigned facei = prim->getComponentIndex();
	Facet *face = mesh->getFace(facei);
	
	for(int i = 0; i < 3; i++) {
		Vertex * v = face->vertex(i);
		if(!isVertexSelected(v->getIndex())) {
			m_vertices.push_back(v);
		}
	}
}

unsigned SelectionArray::numPrims() const
{
	return (unsigned)m_prims.size();
}

Primitive * SelectionArray::getPrimitive(const unsigned & idx) const
{
	return m_prims[idx];
}

unsigned SelectionArray::numVertices() const
{
	return (unsigned)m_vertices.size();
}

Vertex * SelectionArray::getVertex(const unsigned & idx) const
{
	return m_vertices[idx];
}

bool SelectionArray::isVertexSelected(unsigned idx) const
{
	std::vector<Vertex *>::const_iterator vIt;
	for(vIt = m_vertices.begin(); vIt != m_vertices.end(); ++vIt) {
		if((*vIt)->getIndex() == idx)
			return true;
	}
	return false;
}

