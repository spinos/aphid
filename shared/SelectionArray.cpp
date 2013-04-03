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
	m_faceIds.clear();
}

void SelectionArray::add(Geometry * geo, unsigned icomp)
{
    m_geometry = geo;
	MeshLaplacian * mesh = (MeshLaplacian *)geo;
	if(getComponentFilterType() == PrimitiveFilter::TFace) {
	    if(isFaceSelected(icomp)) return;
	    m_faceIds.push_back(icomp);
	    
	    Facet *face = mesh->getFace(icomp);
	
        for(int i = 0; i < 3; i++) {
            Vertex * v = face->vertex(i);
            if(!isVertexSelected(v->getIndex())) {
                m_vertices.push_back(v);
            }
        }
    }
    else {
        Vertex * v = mesh->getVertex(icomp);
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

Vector3F * SelectionArray::getVertexP(const unsigned & idx) const
{
    return m_vertices[idx]->m_v;
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
