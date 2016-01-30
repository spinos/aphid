/*
 *  PlantSelection.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 1/31/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "PlantSelection.h"

namespace sdb {

PlantSelection::PlantSelection(WorldGrid<Array<int, Plant>, Plant > * grid)
{ 
	m_grid = grid; 
	m_plants = new Array<int, PlantInstance>();
	m_numSelected = 0;
}

PlantSelection::~PlantSelection()
{ delete m_plants; }
	
void PlantSelection::set(const Vector3F & center, const Vector3F & direction,
		const float & radius)
{
	m_center = center;
	m_direction = direction;
	m_radius = radius;
}

void PlantSelection::select(SelectionContext::SelectMode mode)
{
	int ng = 1 + m_radius / m_grid->gridSize();
	
	const Coord3 c0 = m_grid->gridCoord((const float *)&m_center);
	Coord3 c1;
	int i, j, k;
	for(k=-ng; k<=ng; ++k) {
		c1.z = c0.z + k;
		for(j=-ng; j<=ng; ++j) {
			c1.y = c0.y + j;
			for(i=-ng; i<=ng; ++i) {
				c1.x = c0.x + i;
				select(c1, mode);
			}
		}
	}
	m_numSelected = m_plants->size();
}

void PlantSelection::select(const Coord3 & c, SelectionContext::SelectMode mode)
{
	BoundingBox b = m_grid->coordToGridBBox(c);
	if(!b.isPointAround(m_center, m_radius) ) return;
	Array<int, Plant> * cell = m_grid->findCell(c);
	if(!cell) return;
	if(cell->isEmpty() ) return;
	cell->begin();
	while(!cell->end()) {
		PlantData * d = cell->value()->index;
		if(m_center.distanceTo(d->t1->getTranslation() ) < m_radius) {
			if(mode == SelectionContext::Append) select(cell->value() );
			else if(mode == SelectionContext::Remove) m_plants->remove(cell->key() );
		}
		
		cell->next();
	}
}

void PlantSelection::select(Plant * p)
{ 
	PlantData * backup = new PlantData;
	*backup->t1 = *p->index->t1;
	*backup->t2 = *p->index->t2;
	*backup->t3 = *p->index->t3;
	
	PlantInstance * inst = new PlantInstance;
	inst->m_backup = backup;
	inst->m_reference = p->index;
	m_plants->insert(p->key, inst);
}

void PlantSelection::deselect()
{ 
	m_plants->clear();
	m_numSelected = 0;
}

const unsigned & PlantSelection::count() const
{ return m_numSelected; }
	
Array<int, PlantInstance> * PlantSelection::data()
{ return m_plants; }

void PlantSelection::calculateWeight()
{
	m_plants->begin();
	while(!m_plants->end() ) {
		const Vector3F pr = m_plants->value()->m_reference->t1->getTranslation();
		const float dist = m_center.distanceTo(pr);
		if(dist < m_radius) {
			m_plants->value()->m_weight = 1.f - dist / m_radius;
		}
		else {
			m_plants->value()->m_weight = 0.f;
		}
		m_plants->next();
	}
}

}