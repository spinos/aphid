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
BuildKdTreeContext::~BuildKdTreeContext() 
{
	m_primitives.clear();
	m_indices.clear();
	m_nodes.clear();
}

void BuildKdTreeContext::appendMesh(BaseMesh* mesh)
{
	unsigned numFace = mesh->getNumFaces();
	m_primitives.expandBy(numFace);
	m_indices.expandBy(numFace);
	
	for(unsigned i = 0; i < numFace; i++) {
		Primitive *p = m_primitives.asPrimitive();
		p->setGeometry((char *)mesh);
		p->setComponentIndex(i);
		unsigned *dest = m_indices.asIndex();
		*dest = m_primitives.index();
		m_primitives.next();
		m_indices.next();
	}
	printf("primitive state:\n");
	m_primitives.verbose();
	printf("indices state:\n");
	m_indices.verbose();
}

const unsigned BuildKdTreeContext::getNumPrimitives() const
{
	return m_primitives.index();
}

const PrimitiveArray &BuildKdTreeContext::getPrimitives() const
{
	return m_primitives;
}

const IndexArray &BuildKdTreeContext::getIndices() const
{
	return m_indices;
}

PrimitiveArray &BuildKdTreeContext::primitives()
{
	return m_primitives;
}
	
IndexArray &BuildKdTreeContext::indices()
{
	return m_indices;
}

KdTreeNode *BuildKdTreeContext::createTreeBranch()
{
	KdTreeNode *p = (KdTreeNode *)m_nodes.expandBy(2);
	unsigned long * tmp = (unsigned long*)m_nodes.current();
	tmp[1] = tmp[3] = 6;
	m_nodes.next();
	tmp = (unsigned long*)m_nodes.current();
	tmp[1] = tmp[3] = 6;
	m_nodes.next();
	return p;
}

KdTreeNode *BuildKdTreeContext::firstTreeBranch()
{
	return m_nodes.asKdTreeNode(0);
}

void BuildKdTreeContext::releaseAt(unsigned loc)
{
	m_indices.shrinkTo(loc);
	m_indices.setIndex(loc);
	m_primitives.shrinkTo(loc);
	m_primitives.setIndex(loc);
}

void BuildKdTreeContext::createPrimitiveBoxes()
{
	const unsigned numPrim = getNumPrimitives();
	m_primitiveBoxes = new BoundingBox[numPrim];
	m_primitives.begin();
	for(unsigned i = 0; i < numPrim; i++) {
		Primitive *p = m_primitives.asPrimitive();
		BaseMesh *mesh = (BaseMesh *)(p->getGeometry());
		unsigned triIdx = p->getComponentIndex();
		
		m_primitiveBoxes[i] = mesh->calculateBBox(triIdx);
		m_primitives.next();
	}
}

void BuildKdTreeContext::clearPrimitiveBoxes()
{
	delete[] m_primitiveBoxes;
}

void BuildKdTreeContext::verbose() const
{
	printf("primitives state:\n");
	m_primitives.verbose();
	printf("indices state:\n");
	m_indices.verbose();
	printf("nodes state:\n");
	m_nodes.verbose();
}