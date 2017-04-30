/*
 *  WorldGrid3.h
 *  
 *  highest level of spatial division, using grid(x,y,z) as key
 *  cell centered at coord point
 *  fixed cell size
 *
 *  Created by jian zhang on 3/8/17.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SDB_WORLD_GRID_3_H
#define APH_SDB_WORLD_GRID_3_H

#include <sdb/Sequence.h>
#include <math/BoundingBox.h>

namespace aphid {

namespace sdb {

template<typename ChildType>
class WorldGrid3 : public Sequence<Coord3>
{
	BoundingBox m_bbox;
	static const float m_cellSize;
	static const float m_halfCellSize;
	
public:
	WorldGrid3(Entity * parent = NULL);
	virtual ~WorldGrid3();
	
	ChildType * value();
	const Coord3 & key() const;

	ChildType * insertCell(const float * at);
	ChildType * insertCell(const Coord3 & x);
	
/// find cell coord of point(x,y,z)
	const Coord3 cellCoord(const float * p) const;

	void calculateBBox();
	const BoundingBox coordToCellBBox(const Coord3 & c) const;
	Vector3F coordToCellCenter(const Coord3 & c) const;
	const BoundingBox & boundingBox() const;

	ChildType * findCell(const Vector3F & pref);
	ChildType * findCell(const Coord3 & c);
	
	void getCellBBox(BoundingBox & bx, const Coord3 & c);
	
	static int TwentySixNeighborCoord[26][3];
	
protected:
	const BoundingBox keyToCellBBox() const;
	
private:
	
};

template<typename ChildType>
const float WorldGrid3<ChildType>::m_cellSize = 2048.f;

template<typename ChildType>
const float WorldGrid3<ChildType>::m_halfCellSize = 1024.f;

template<typename ChildType>
WorldGrid3<ChildType>::WorldGrid3(Entity * parent) : Sequence<Coord3>(parent)
{}

template<typename ChildType>
WorldGrid3<ChildType>::~WorldGrid3() 
{}

template<typename ChildType>
ChildType * WorldGrid3<ChildType>::value() 
{ 
	ChildType * r = dynamic_cast<ChildType *>(Sequence<Coord3>::currentIndex() );
	if(r == NULL) {
		throw "WorldGrid3 value is null ";
	}
	return r;
}

template<typename ChildType>
const Coord3 & WorldGrid3<ChildType>::key() const 
{ return Sequence<Coord3>::currentKey(); }

template<typename ChildType>
const Coord3 WorldGrid3<ChildType>::cellCoord(const float * p) const
{
	Vector3F vp(p);
	vp.x += m_halfCellSize;
	vp.y += m_halfCellSize;
	vp.z += m_halfCellSize;
	
	Coord3 r;
	r.x = vp.x / m_cellSize; if(vp.x < 0.f) r.x--;
	r.y = vp.y / m_cellSize; if(vp.y < 0.f) r.y--;
	r.z = vp.z / m_cellSize; if(vp.z < 0.f) r.z--;
	return r;
}

template<typename ChildType>
ChildType * WorldGrid3<ChildType>::insertCell(const float * at)
{
	const Coord3 x = cellCoord(at);
	return insertCell(x);
}

template<typename ChildType>
ChildType * WorldGrid3<ChildType>::insertCell(const Coord3 & x)
{
	Pair<Coord3, Entity> * p = Sequence<Coord3>::insert(x);
	if(!p->index) {
		p->index = new ChildType(this);
	}
	
	return static_cast<ChildType *>(p->index);
}

template<typename ChildType>
void WorldGrid3<ChildType>::calculateBBox()
{
	m_bbox.reset();
	begin();
	while(!end()) {
		m_bbox.expandBy(coordToCellBBox( key() ));
		next();
	}
}

template<typename ChildType>
const BoundingBox WorldGrid3<ChildType>::coordToCellBBox(const Coord3 & c) const
{
	BoundingBox bb;
	bb.setMin(m_cellSize * c.x - m_halfCellSize, m_cellSize * c.y - m_halfCellSize, m_cellSize * c.z - m_halfCellSize );
	bb.setMax(m_cellSize * c.x + m_halfCellSize, m_cellSize * c.y + m_halfCellSize, m_cellSize * c.z + m_halfCellSize );
	return bb;
}

template<typename ChildType>
Vector3F WorldGrid3<ChildType>::coordToCellCenter(const Coord3 & c) const
{
	return Vector3F(m_cellSize * c.x, 
	                m_cellSize * c.y, 
	                m_cellSize * c.z);
}

template<typename ChildType>
const BoundingBox & WorldGrid3<ChildType>::boundingBox() const
{ return m_bbox; }

template<typename ChildType>
ChildType * WorldGrid3<ChildType>::findCell(const Vector3F & pref)
{
	Coord3 c0 = gridCoord((float *)&pref);
	return findCell(c0);
}

template<typename ChildType>
ChildType * WorldGrid3<ChildType>::findCell(const Coord3 & c)
{
	Pair<Entity *, Entity> p = findEntity(c);
	if(p.index) {
		ChildType * g = static_cast<ChildType *>(p.index);
		return g;
	}
	return NULL;
}

template<typename ChildType>
const BoundingBox WorldGrid3<ChildType>::keyToCellBBox() const
{ return coordToCellBBox(key() ); }

template<typename ChildType>
void WorldGrid3<ChildType>::getCellBBox(BoundingBox & bx, const Coord3 & c)
{ bx = coordToCellBBox(c); }

template<typename ChildType>
int WorldGrid3<ChildType>::TwentySixNeighborCoord[26][3] = {
{-1, 0, 0}, // face
{ 1, 0, 0},
{ 0,-1, 0},
{ 0, 1, 0},
{ 0, 0,-1},
{ 0, 0, 1},
{-1,-1,-1}, // vertex
{ 1,-1,-1},
{-1, 1,-1},
{ 1, 1,-1},
{-1,-1, 1},
{ 1,-1, 1},
{-1, 1, 1},
{ 1, 1, 1},
{-1, 0,-1}, // edge
{ 1, 0,-1},
{-1, 0, 1},
{ 1, 0, 1},
{ 0,-1,-1},
{ 0, 1,-1},
{ 0,-1, 1},
{ 0, 1, 1},
{-1,-1, 0},
{ 1,-1, 0},
{-1, 1, 0},
{ 1, 1, 0}
};

} //end namespace sdb

}
#endif