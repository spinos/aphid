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

namespace aphid {

BuildKdTreeContext * BuildKdTreeContext::GlobalContext = NULL;

BuildKdTreeContext::BuildKdTreeContext() : m_grid(NULL),
m_numPrimitive(0) {}

BuildKdTreeContext::BuildKdTreeContext(BuildKdTreeStream &data, const BoundingBox & b)
{
	setBBox(b);
	m_grid = new GridClustering();
	m_grid->setGridSize(b.getLongestDistance() / 64.f);
	std::cout<<"\n ctx grid size "<<m_grid->gridSize();
	
	m_numPrimitive = data.getNumPrimitives();
	std::cout<<"\n n prims "<<m_numPrimitive;
	
	int igeom, icomp;
	const sdb::VectorArray<Primitive> &primitives = data.primitives();
	
	for(unsigned i = 0; i < m_numPrimitive; i++) {
				
		Primitive *p = primitives.get(i);
		
		p->getGeometryComponent(igeom, icomp);

		BoundingBox ab = data.calculateComponentBox(igeom, icomp);
		
		ab.expand(1e-6f);
		
		m_primitiveBoxes.insert(ab);
		
        m_indices.insert(i);
		
		m_grid->insertToGroup(ab, i);
		
	}
	
	std::cout<<"\n ctx grid n cell "<<m_grid->size();
}

BuildKdTreeContext::~BuildKdTreeContext() 
{
	if(m_grid) delete m_grid;
}

void BuildKdTreeContext::createGrid(const float & x)
{
	m_grid = new GridClustering();
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

const sdb::VectorArray<unsigned> & BuildKdTreeContext::indices() const
{
	return m_indices;
}

const sdb::VectorArray<BoundingBox> & BuildKdTreeContext::primitiveBoxes() const
{
	return m_primitiveBoxes;
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

GridClustering * BuildKdTreeContext::grid()
{ return m_grid; }

void BuildKdTreeContext::addCell(const sdb::Coord3 & x, GroupCell * c)
{ m_grid->insertChildValue(x, c); }

void BuildKdTreeContext::countPrimsInGrid()
{
	m_numPrimitive = 0;
	if(!m_grid) return;
	m_numPrimitive = m_grid->numElements();
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
/// reset
        m_indices.clear();
		m_numPrimitive = 0;
		
		const sdb::VectorArray<BoundingBox> & boxSrc = GlobalContext->primitiveBoxes();
		m_grid->extractInside(m_indices, boxSrc, m_bbox );
		
		m_numPrimitive = m_indices.size();
		
		delete m_grid;
		m_grid = NULL;
		
		return true;
	}
	return false;
}

void BuildKdTreeContext::addIndex(const unsigned & x)
{
	m_indices.insert( x);
	m_numPrimitive++;
}

}