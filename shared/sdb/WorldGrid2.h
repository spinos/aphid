/*
 *  WorldGrid2.h
 *  
 *  world grid without value insertion
 *  highest level of spatial division, using grid(x,y,z) as key
 *
 *  Created by jian zhang on 1/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SDB_WORLD_GRID_2_H
#define APH_SDB_WORLD_GRID_2_H
#include <sdb/Sequence.h>
#include <math/BoundingBox.h>

namespace aphid {

namespace sdb {

template<typename ChildType>
class WorldGrid2 : public Sequence<Coord3>
{
	BoundingBox m_bbox;
/// size of cell
	float m_gridSize;
	
public:
	WorldGrid2(Entity * parent = NULL);
	
	virtual ~WorldGrid2();
	
	void setGridSize(float x);
	
	ChildType * value();
	
	const Coord3 & key() const;

	ChildType * insertCell(const float * at);
	ChildType * insertCell(const Coord3 & x);
	
/// find grid coord of point(x,y,z)
	const Coord3 gridCoord(const float * p) const;

	void calculateBBox();
	const BoundingBox coordToCellBBox(const Coord3 & c) const;
	Vector3F coordToCellCenter(const Coord3 & c) const;
	const BoundingBox & boundingBox() const;

	const float & gridSize() const;

	ChildType * findCell(const Vector3F & pref);
	ChildType * findCell(const Coord3 & c);
	
	void getCellBBox(BoundingBox & bx, const Coord3 & c);
	
	static int TwentySixNeighborCoord[26][3];
	
protected:
	float * gridSizeR();
    BoundingBox * boundingBoxR();
    const BoundingBox keyToCellBBox() const;
	
private:
	
};

template<typename ChildType>
WorldGrid2<ChildType>::WorldGrid2(Entity * parent) : Sequence<Coord3>(parent),
m_gridSize(32.f)
{}

template<typename ChildType>
WorldGrid2<ChildType>::~WorldGrid2() 
{}

template<typename ChildType>
void WorldGrid2<ChildType>::setGridSize(float x)
{ m_gridSize = (x>32.f) ? x : 32.f; }

template<typename ChildType>
ChildType * WorldGrid2<ChildType>::value() 
{ 
	ChildType * r = dynamic_cast<ChildType *>(Sequence<Coord3>::currentIndex() );
	if(r == NULL) {
		throw "WorldGrid22 value is null ";
	}
	return r;
}

template<typename ChildType>
const Coord3 & WorldGrid2<ChildType>::key() const 
{ return Sequence<Coord3>::currentKey(); }
	

template<typename ChildType>
const Coord3 WorldGrid2<ChildType>::gridCoord(const float * p) const
{
	Coord3 r;
	r.x = p[0] / m_gridSize; if(p[0] < 0.f) r.x--;
	r.y = p[1] / m_gridSize; if(p[1] < 0.f) r.y--;
	r.z = p[2] / m_gridSize; if(p[2] < 0.f) r.z--;
	return r;
}

template<typename ChildType>
ChildType * WorldGrid2<ChildType>::insertCell(const float * at)
{
	const Coord3 x = gridCoord(at);
	return insertCell(x);
}

template<typename ChildType>
ChildType * WorldGrid2<ChildType>::insertCell(const Coord3 & x)
{
	Pair<Coord3, Entity> * p = Sequence<Coord3>::insert(x);
	if(!p->index) {
		p->index = new ChildType(this);
	}
	
	return static_cast<ChildType *>(p->index);
}

template<typename ChildType>
void WorldGrid2<ChildType>::calculateBBox()
{
	m_bbox.reset();
	begin();
	while(!end()) {
		m_bbox.expandBy(coordToGridBBox( key() ));
		next();
	}
}

template<typename ChildType>
const BoundingBox WorldGrid2<ChildType>::coordToCellBBox(const Coord3 & c) const
{
	BoundingBox bb;
	bb.setMin(m_gridSize * c.x, m_gridSize * c.y, m_gridSize * c.z);
	bb.setMax(m_gridSize * (c.x + 1), m_gridSize * (c.y + 1), m_gridSize * (c.z + 1));
	return bb;
}

template<typename ChildType>
Vector3F WorldGrid2<ChildType>::coordToCellCenter(const Coord3 & c) const
{
	return Vector3F(m_gridSize * (.5f + c.x), 
	                m_gridSize * (.5f + c.y), 
	                m_gridSize * (.5f + c.z) );
}

template<typename ChildType>
const BoundingBox & WorldGrid2<ChildType>::boundingBox() const
{ return m_bbox; }

template<typename ChildType>
const float & WorldGrid2<ChildType>::gridSize() const
{ return m_gridSize; }

template<typename ChildType>
ChildType * WorldGrid2<ChildType>::findCell(const Vector3F & pref)
{
	Coord3 c0 = gridCoord((float *)&pref);
	return findCell(c0);
}

template<typename ChildType>
ChildType * WorldGrid2<ChildType>::findCell(const Coord3 & c)
{
	Pair<Entity *, Entity> p = findEntity(c);
	if(p.index) {
		ChildType * g = static_cast<ChildType *>(p.index);
		return g;
	}
	return NULL;
}

template<typename ChildType>
float * WorldGrid2<ChildType>::gridSizeR()
{ return &m_gridSize; }

template<typename ChildType>
BoundingBox * WorldGrid2<ChildType>::boundingBoxR()
{ return &m_bbox; }

template<typename ChildType>
const BoundingBox WorldGrid2<ChildType>::keyToCellBBox() const
{ return coordToCellBBox(key() ); }

template<typename ChildType>
void WorldGrid2<ChildType>::getCellBBox(BoundingBox & bx, const Coord3 & c)
{ bx = coordToCellBBox(c); }

template<typename ChildType>
int WorldGrid2<ChildType>::TwentySixNeighborCoord[26][3] = {
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