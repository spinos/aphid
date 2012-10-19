/*
 *  BuildKdTreeContext.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BuildKdTreeContext.h"
BuildKdTreeContext::BuildKdTreeContext() {}

void BuildKdTreeContext::appendMesh(BaseMesh* mesh)
{
	unsigned numFace = mesh->getNumFaces();
	unsigned offset = m_primitives.size();
	m_primitives.allocate(numFace);
	for(unsigned i = 0; i < numFace; i++) {
		m_primitives[offset + i] = *mesh->getFace(i);
		m_primitives[offset + i].setType(0);
		
		Triangle *tri = (Triangle *)mesh->getFace(i);
		tri->expandBBox(m_bbox);
	}
}

const unsigned BuildKdTreeContext::getNumPrimitives() const
{
	return m_primitives.size();
}

const BoundingBox BuildKdTreeContext::getBBox() const
{
	return m_bbox;
}