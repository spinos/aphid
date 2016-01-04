/*
 *  WorldGrid.h
 *  
 *
 *  Created by jian zhang on 1/2/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 *  Highest level of spatial division, using grid(x,y,z) as key
 */

#pragma once
#include <Sequence.h>
namespace sdb {

template<typename ChildType, typename ValueType>
class WorldGrid : public Sequence<Coord3>
{
	float m_gridSize;
	
public:
	WorldGrid(Entity * parent = NULL) : Sequence<Coord3>(parent) 
	{
		m_gridSize = 1.f;
	}
	
	virtual ~WorldGrid() {}
	
	void setGridSize(float x)
	{ m_gridSize = x; }
	
	ChildType * value() 
	{ return static_cast<ChildType *>(Sequence<Coord3>::currentIndex() ); }
	
	const Coord3 key() const 
	{ return Sequence<Coord3>::currentKey(); }
	
/// put v into grid
	void insert(const float * at, const ValueType & v);

protected:	
/// find grid coord of point(x,y,z)
	const Coord3 gridCoord(const float * p) const;
	
private:
	
};

template<typename ChildType, typename ValueType>
const Coord3 WorldGrid<ChildType, ValueType>::gridCoord(const float * p) const
{
	Coord3 r;
	r.x = p[0] / m_gridSize; if(p[0] < 0.f) r.x--;
	r.y = p[1] / m_gridSize; if(p[1] < 0.f) r.y--;
	r.z = p[2] / m_gridSize; if(p[2] < 0.f) r.z--;
	return r;
}

template<typename ChildType, typename ValueType>
void WorldGrid<ChildType, ValueType>::insert(const float * at, const ValueType & v) 
{
	const Coord3 x = gridCoord(at);
	
	Pair<Coord3, Entity> * p = Sequence<Coord3>::insert(x);
	if(!p->index) p->index = new ChildType(this);
	static_cast<ChildType *>(p->index)->insert(v);
}

} //end namespace sdb
