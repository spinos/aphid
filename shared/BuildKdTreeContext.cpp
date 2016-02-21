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

BuildKdTreeContext::BuildKdTreeContext() : m_grid(NULL),
m_numPrimitive(0) {}

BuildKdTreeContext::BuildKdTreeContext(BuildKdTreeStream &data, const BoundingBox & b)
{
	setBBox(b);
	m_grid = new sdb::WorldGrid<GroupCell, unsigned >();
	m_grid->setGridSize(b.getLongestDistance() / 64.f);
	std::cout<<"\n ctx grid size "<<m_grid->gridSize();
	
	m_numPrimitive = data.getNumPrimitives();
	std::cout<<"\n n prims "<<m_numPrimitive;
	
	createIndirection(m_numPrimitive);
	
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
		
		// primBoxes[i].expand(1e-6f);	
		
		const Vector3F center = primBoxes[i].center();
		
		*primIndex = i;
		primIndex++;
		
		GroupCell * c = m_grid->insertChild((const float *)&center);
		
		if(!c) {
			std::cout<<"\n error cast to GroupCell";
			return;
		}
		
		c->insert(i);
		c->m_box.expandBy(primBoxes[i]);
	}
	
	std::cout<<"\n ctx grid n cell "<<m_grid->size();
}

BuildKdTreeContext::~BuildKdTreeContext() 
{
	if(m_grid) delete m_grid;
}

/// allocate by max count
void BuildKdTreeContext::createIndirection(const unsigned &count)
{ m_indices.create(count+1); }

void BuildKdTreeContext::createGrid(const float & x)
{
	m_grid = new sdb::WorldGrid<GroupCell, unsigned >();
	m_grid->setGridSize(x);
	m_grid->setDataExternal();
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

bool BuildKdTreeContext::isCompressed()
{ return m_grid !=NULL; }

sdb::WorldGrid<GroupCell, unsigned > * BuildKdTreeContext::grid()
{ return m_grid; }

void BuildKdTreeContext::addCell(const sdb::Coord3 & x, GroupCell * c)
{ m_grid->insertChildValue(x, c); }

void BuildKdTreeContext::countPrimsInGrid()
{
	m_numPrimitive = 0;
	if(!m_grid) return;
	m_grid->begin();
	while(!m_grid->end() ) {
		m_numPrimitive += m_grid->value()->size();
		m_grid->next();
	}
}

int BuildKdTreeContext::numCells()
{
	if(!m_grid) return 0;
	return m_grid->size();
}

bool BuildKdTreeContext::decompress(bool forced)
{
	if(!m_grid) return false;
	if(m_numPrimitive < 1024 
		|| numCells() < 32
		|| forced) {
		m_indices.create(m_numPrimitive+1);
		unsigned ind = 0;
		m_grid->begin();
		while(!m_grid->end() ) {
			addIndices(m_grid->value(), ind );
			m_grid->next();
		}
		delete m_grid;
		m_grid = NULL;
		return true;
	}
	return false;
}

void BuildKdTreeContext::addIndices(GroupCell * c, unsigned &ind)
{
	c->begin();
	while(!c->end() ) {
		m_indices.ptr()[ind] = c->key();
		ind++;
		c->next();
	}
}

void BuildKdTreeContext::addIndex(const unsigned & x)
{
	m_indices.ptr()[m_numPrimitive] = x;
	m_numPrimitive++;
}
