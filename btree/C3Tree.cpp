/*
 *  C3Tree.cpp
 *  C3Tree
 *
 *  Created by jian zhang on 5/5/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "C3Tree.h"

namespace sdb {

C3Tree::C3Tree(Entity * parent) : Ordered<Coord3, VertexP>(parent)
{
	m_gridSize = 1.f;
}

void C3Tree::setGridSize(const float & x) { m_gridSize = x; }

void C3Tree::insert(const VertexP & v)
{
	Coord3 k = inGrid(*v.index);
	
	Pair<Coord3, Entity> * p = Sequence<Coord3>::insert(k);
	if(!p->index) p->index = new List<VertexP>;
	static_cast<List<VertexP> *>(p->index)->insert(v);
}

void C3Tree::remove(const VertexP & v)
{
	Coord3 k = inGrid(*v.index);
	Sequence<Coord3>::remove(k);
}

List<VertexP> * C3Tree::find(float * p)
{
	V3 v(p);
	
	Coord3 k = inGrid(v);
	
	Pair<Entity *, Entity> g = Sequence<Coord3>::findEntity(k);
	if(!g.index) return NULL;
	return static_cast<List<VertexP> *>(g.index);
}

const Coord3 C3Tree::inGrid(const V3 & p) const
{
	return gridCoord(p.data);
}

const Coord3 C3Tree::gridCoord(const float * p) const
{
	Coord3 r;
	r.x = p[0] / m_gridSize; if(p[0] < 0.f) r.x--;
	r.y = p[1] / m_gridSize; if(p[1] < 0.f) r.y--;
	r.z = p[2] / m_gridSize; if(p[2] < 0.f) r.z--;
	return r;
}

const BoundingBox C3Tree::gridBoundingBox() const
{
	return coordToGridBBox(Sequence<Coord3>::currentKey());
}

const BoundingBox C3Tree::boundingBox() const
{
	return m_bbox;
}

void C3Tree::calculateBBox()
{
	m_bbox.reset();
	begin();
	while(!end()) {
		updateBBox(gridBoundingBox());
		next();
	}
}

const BoundingBox C3Tree::coordToGridBBox(const Coord3 & c) const
{
	BoundingBox bb;
	bb.setMin(m_gridSize * c.x, m_gridSize * c.y, m_gridSize * c.z);
	bb.setMax(m_gridSize * (c.x + 1), m_gridSize * (c.y + 1), m_gridSize * (c.z + 1));
	return bb;
}

void C3Tree::updateBBox(const BoundingBox & b)
{
	m_bbox.expandBy(b);
}

const List<VertexP> * C3Tree::verticesInGrid() const
{
	Entity * p = Sequence<Coord3>::currentIndex();
	if(!p) return NULL;
	return static_cast<List<VertexP> *>(p);
}

const float C3Tree::gridSize() const { return m_gridSize; }

} // end namespace sdb