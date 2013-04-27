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
#include "BaseMesh.h"
#include <VertexPath.h>
#include <VertexAdjacency.h>

SelectionArray::SelectionArray() 
{
	m_vertexPath = new VertexPath;
}

SelectionArray::~SelectionArray() {}

void SelectionArray::reset() 
{
	m_prims.clear();
	m_vertexIds.clear();
	m_faceIds.clear();
}

void SelectionArray::add(Geometry * geo, unsigned icomp)
{
    m_geometry = geo;
	BaseMesh * mesh = (BaseMesh *)geo;
	if(getComponentFilterType() == PrimitiveFilter::TFace) {
	    if(isFaceSelected(icomp)) return;
	    m_faceIds.push_back(icomp);
	    
	    unsigned vertexId[3];
		mesh->getTriangle(icomp, vertexId);
	
        for(int i = 0; i < 3; i++)
            addVertex(vertexId[i]);
    }
    else {
		if(isVertexSelected(icomp)) return;
		
		if(numVertices() < 1) {
			m_vertexIds.push_back(icomp);
			return;
		}
		
		if(!m_needVertexPath) {
			addVertex(icomp);
			return;
		}
		
		unsigned startVert = lastVertexId();
		unsigned endVert = icomp;
		m_vertexPath->create(startVert, endVert);
		
		for(unsigned i = 0; i < m_vertexPath->numVertices(); i++)
			addVertex(m_vertexPath->vertex(i));
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
	return (unsigned)m_vertexIds.size();
}

unsigned SelectionArray::getVertexId(const unsigned & idx) const
{
	return m_vertexIds[idx];
}

unsigned SelectionArray::lastVertexId() const
{
	return m_vertexIds[m_vertexIds.size() - 1];
}

Vector3F * SelectionArray::getVertexP(const unsigned & idx) const
{
	BaseMesh * mesh = (BaseMesh *)m_geometry;
    return &mesh->getVertices()[m_vertexIds[idx]];
}

bool SelectionArray::isVertexSelected(unsigned idx) const
{
	std::vector<unsigned>::const_iterator vIt;
	for(vIt = m_vertexIds.begin(); vIt != m_vertexIds.end(); ++vIt) {
		if((*vIt) == idx)
			return true;
	}
	return false;
}

void SelectionArray::addVertex(unsigned idx)
{
	if(!isVertexSelected(idx))
		m_vertexIds.push_back(idx);
}

bool SelectionArray::isFaceSelected(unsigned idx) const
{
	std::vector<unsigned>::const_iterator it;
	for(it = m_faceIds.begin(); it != m_faceIds.end(); ++it) {
		if((*it) == idx)
			return true;
	}
	return false;
}

Geometry * SelectionArray::getGeometry() const
{
    return m_geometry;
}

unsigned SelectionArray::numFaces() const
{
    return (unsigned)m_faceIds.size();
}

unsigned SelectionArray::getFaceId(const unsigned & idx) const
{
    return m_faceIds[idx];
}

void SelectionArray::setTopology(VertexAdjacency * topo)
{
	m_vertexPath->setTopology(topo);
}

void SelectionArray::grow()
{
	if(numVertices() < 2) return;
	
	std::vector<unsigned>::iterator it = m_vertexIds.end();
	it--;
	unsigned endVert = *it;
	it--;
	unsigned startVert = *it;
	unsigned nextVert;
	if(m_vertexPath->grow(startVert, endVert, nextVert))
		addVertex(nextVert);	
}

void SelectionArray::shrink()
{
	if(numVertices() > 0) {
		std::vector<unsigned>::iterator it = m_vertexIds.end();
		--it;
		m_vertexIds.erase(it);
	}
}

void SelectionArray::enableVertexPath()
{
	m_needVertexPath = true;
}

void SelectionArray::disableVertexPath()
{
	m_needVertexPath = false;
}
//:~
