/*
 *  BuildKdTreeContext.cpp
 *  kdtree
 *
 *  Created by jian zhang on 10/20/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include "BuildKdTreeContext.h"
#include "Geometry.h"
#include <VectorArray.h>

BuildKdTreeContext::BuildKdTreeContext() : m_grid(NULL) {}

BuildKdTreeContext::BuildKdTreeContext(BuildKdTreeStream &data, const BoundingBox & b)
{
	setBBox(b);
	m_grid = new sdb::WorldGrid<GroupCell, unsigned >();
	m_grid->setGridSize(b.getLongestDistance() / 60.f);
	std::cout<<"\n ctx grid size "<<m_grid->gridSize();
	
	createIndirection(data.getNumPrimitives());
/// copy bbox of all prims
	m_primitiveBoxes.create(m_numPrimitive+1);
	
	BoundingBox *primBoxes = m_primitiveBoxes.ptr();
	unsigned *primIndex = m_indices.ptr();
	
	sdb::VectorArray<Primitive> &primitives = data.primitives();
	
	for(unsigned i = 0; i < m_numPrimitive; i++) {
				
		Primitive *p = primitives.get(i);

		Geometry *geo = p->getGeometry();
		
		unsigned compIdx = p->getComponentIndex();
		
		primBoxes[i] = geo->calculateBBox(compIdx);
		primBoxes[i].expand(1e-6f);	
		
		const Vector3F center = primBoxes[i].center();
		
		*primIndex = i;
		primIndex++;
		
		GroupCell * c = m_grid->insertChild((const float *)&center);
		c->insert(i);
		c->m_box.expandBy(primBoxes[i]);
	}
	
	std::cout<<"\n ctx grid n cell "<<m_grid->size();
}

BuildKdTreeContext::~BuildKdTreeContext() 
{
	if(m_grid) delete m_grid;
}

void BuildKdTreeContext::createIndirection(const unsigned &count)
{
	m_numPrimitive = count;
	m_indices.create(m_numPrimitive+1);
}

void BuildKdTreeContext::setBBox(const BoundingBox &bbox)
{
	m_bbox = bbox;
}

const BoundingBox & BuildKdTreeContext::getBBox() const
{
	return m_bbox;
}

const unsigned & BuildKdTreeContext::getNumPrimitives() const
{
	return m_numPrimitive;
}

unsigned *BuildKdTreeContext::indices()
{
	return m_indices.ptr();
}

BoundingBox *BuildKdTreeContext::primitiveBoxes()
{
	return m_primitiveBoxes.ptr();
}

float BuildKdTreeContext::visitCost() const
{
	return 2.f * m_numPrimitive;
}

void BuildKdTreeContext::verbose() const
{
	//printf("indices state:\n");
	//m_indices.verbose();
}