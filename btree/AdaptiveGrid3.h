/*
 *  AdaptiveGrid3.h
 *  
 *	key (x,y,z,level)
 *  coarsest level_0 cell size 2^max_level*h
 *  coord(2^max_level*n, 2^max_level*n, 2^max_level*n, 0)
 *  finest leve_max_level-1 cell size h
 *  coord(n, n, n, max_level-1)
 *
 *  Created by jian zhang on 7/21/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Sequence.h>
#include <BoundingBox.h>
#include "GridTables.h"

namespace aphid {

namespace sdb {

template<typename CellType, typename ValueType, int MaxLevel = 4>
class AdaptiveGrid3 : public Sequence<Coord4>
{
	BoundingBox m_bbox;
/// size of cell of each level
	float m_cellSize[MaxLevel+1];
	int m_levelCoord[MaxLevel+1];
	
public:
	AdaptiveGrid3(Entity * parent = NULL) : Sequence<Coord4>(parent) 
	{}
	
	virtual ~AdaptiveGrid3() 
	{}
	
	int maxLevel() const;
	void setFinestCellSize(const float & x);
	const float & finestCellSize() const;
	const float & coarsestCellSize() const;
	const float & levelCellSize(const int & x) const;
	
	CellType * value();
	const Coord4 key() const;
	
/// add cell at level
	CellType * addCell(const Vector3F & pref, const int & level = 0);
	
	const Coord4 cellCoordAtLevel(const Vector3F & pref, 
						int level) const;
	
/// bbox of all level0 cells
	void calculateBBox();
	const BoundingBox & boundingBox() const;
	void getCellBBox(BoundingBox & b, 
						const Coord4 & c) const;
/// i 0:7
	void getCellChildBox(BoundingBox & b, 
						const int & i,
						const Coord4 & c) const;
	Vector3F cellCenter(const Coord4 & c) const;
	Vector3F cellChildCenter(const Coord4 & c,
						const int & i) const;
						
/// add child i of cell i 0:7
	CellType * subdivide(const Coord4 & cellCoord, const int & i);
	
	CellType * findCell(const Coord4 & c);
	Coord4 parentCoord(const Coord4 & c) const;
/// i 0:25
	Coord4 neighborCoord(const Coord4 & c, const int & i) const;
	
	CellType * findNeighborCell(const Coord4 & c, const int & i);
/// i 0:5 search for level 0 cell	
	bool isCellFaceOnBoundray(const Coord4 & cellCoord, const int & i);
	
	static float CellLevelLegend[12][3];
	
	static void GetCellColor(Vector3F & c, const int & level);
	
protected:
	
private:
	
};

template<typename CellType, typename ValueType, int MaxLevel>
float AdaptiveGrid3<CellType, ValueType, MaxLevel>::CellLevelLegend[12][3] = {
{1.f, 0.f, 0.f},
{0.f, 1.f, 0.f},
{0.f, 0.f, 1.f},
{0.f, 1.f, 1.f},
{1.f, 0.f, 1.f},
{1.f, 1.f, 0.f},
{1.f, .5f, 0.f},
{.5f, 1.f, 0.f},
{.5f, 0.f, 1.f},
{0.f, .5f, .5f},
{.5f, 0.f, .5f},
{.5f, .5f, 0.f}
};

template<typename CellType, typename ValueType, int MaxLevel>
int AdaptiveGrid3<CellType, ValueType, MaxLevel>::maxLevel() const
{ return MaxLevel; }

template<typename CellType, typename ValueType, int MaxLevel>
void AdaptiveGrid3<CellType, ValueType, MaxLevel>::setFinestCellSize(const float & x)
{
	for(int i=0; i<=MaxLevel; ++i) {
		m_levelCoord[i] = 1<<(MaxLevel-i);
		m_cellSize[i] = x * m_levelCoord[i];
	}
}

template<typename CellType, typename ValueType, int MaxLevel>
const float & AdaptiveGrid3<CellType, ValueType, MaxLevel>::finestCellSize() const
{ return m_cellSize[MaxLevel]; }

template<typename CellType, typename ValueType, int MaxLevel>
const float & AdaptiveGrid3<CellType, ValueType, MaxLevel>::coarsestCellSize() const
{ return m_cellSize[0]; }

template<typename CellType, typename ValueType, int MaxLevel>
const float & AdaptiveGrid3<CellType, ValueType, MaxLevel>::levelCellSize(const int & x) const
{ return m_cellSize[x]; }

template<typename CellType, typename ValueType, int MaxLevel>
CellType * AdaptiveGrid3<CellType, ValueType, MaxLevel>::value() 
{ return static_cast<CellType *>(Sequence<Coord4>::currentIndex() ); }

template<typename CellType, typename ValueType, int MaxLevel>
const Coord4 AdaptiveGrid3<CellType, ValueType, MaxLevel>::key() const 
{ return Sequence<Coord4>::currentKey(); }
	
template<typename CellType, typename ValueType, int MaxLevel>
CellType * AdaptiveGrid3<CellType, ValueType, MaxLevel>::addCell(const Vector3F & pref,
														const int & level)
{
	Coord4 c = cellCoordAtLevel(pref, level);
	Pair<Coord4, Entity> * p = Sequence<Coord4>::insert(c);
	if(!p->index)
		p->index = new CellType(this);

	return static_cast<CellType *>(p->index);
}

template<typename CellType, typename ValueType, int MaxLevel>
const Coord4 AdaptiveGrid3<CellType, ValueType, MaxLevel>::cellCoordAtLevel(const Vector3F & pref,
															int level) const
{
	const float & cz = m_cellSize[level];
	Coord4 r;
	r.x = pref.x / cz; if(pref.x < 0.f) r.x--;
	r.y = pref.y / cz; if(pref.y < 0.f) r.y--;
	r.z = pref.z / cz; if(pref.z < 0.f) r.z--;
	r.x *= m_levelCoord[level];
	r.y *= m_levelCoord[level];
	r.z *= m_levelCoord[level];
	r.w = level;
	return r;
}

template<typename CellType, typename ValueType, int MaxLevel>
void AdaptiveGrid3<CellType, ValueType, MaxLevel>::calculateBBox()
{
	const float & cz = m_cellSize[0];
	BoundingBox cb;
	m_bbox.reset();
	begin();
	while(!end()) {
		getCellBBox(cb, key() );
		m_bbox.expandBy(cb );
		
		if(key().w > 0) 
			return;
		
		next();
	}
}

template<typename CellType, typename ValueType, int MaxLevel>
const BoundingBox & AdaptiveGrid3<CellType, ValueType, MaxLevel>::boundingBox() const
{ return m_bbox; }

template<typename CellType, typename ValueType, int MaxLevel>
void AdaptiveGrid3<CellType, ValueType, MaxLevel>::getCellBBox(BoundingBox & b,
											const Coord4 & c) const
{
	const float & cz = finestCellSize();
	const int & gz = m_levelCoord[c.w];
	b.setMin(cz * c.x, cz * c.y, cz * c.z);
	b.setMax(cz * (c.x + gz), cz * (c.y + gz), cz * (c.z + gz) );
	
}

template<typename CellType, typename ValueType, int MaxLevel>
void AdaptiveGrid3<CellType, ValueType, MaxLevel>::GetCellColor(Vector3F & c, 
											const int & level)
{
	c.set(CellLevelLegend[level][0],
			CellLevelLegend[level][1],
			CellLevelLegend[level][2]); 
}

template<typename CellType, typename ValueType, int MaxLevel>
CellType * AdaptiveGrid3<CellType, ValueType, MaxLevel>::subdivide(const Coord4 & cellCoord,
												const int & i)
{
	CellType * cell = findCell(cellCoord);
	if(!cell) {
		std::cout<<"\n [ERROR] cannot find cell to subdivide "<<cellCoord;
		return NULL;
	}
	
	cell->setHasChild();
	const float cz = m_cellSize[cellCoord.w + 1] * .5f;
	return addCell( cellCenter(cellCoord) + Vector3F(gdt::EightCellChildOffset[i][0],
										gdt::EightCellChildOffset[i][1],
										gdt::EightCellChildOffset[i][2]) * cz, cellCoord.w + 1 );
										
}

template<typename CellType, typename ValueType, int MaxLevel>
CellType * AdaptiveGrid3<CellType, ValueType, MaxLevel>::findCell(const Coord4 & c)
{
	Pair<Entity *, Entity> p = findEntity(c);
	if(p.index) {
		CellType * g = static_cast<CellType *>(p.index);
		return g;
	}
	return NULL;
}

template<typename CellType, typename ValueType, int MaxLevel>
void AdaptiveGrid3<CellType, ValueType, MaxLevel>::getCellChildBox(BoundingBox & b, 
						const int & i,
						const Coord4 & c) const
{
	const Vector3F center = cellCenter(c);
	getCellBBox(b, c);
	
	int z = i>>2;
	int y = (i - (z<<2) )>>1;
	int x = i - (z<<2) - (y<<1);
	
	if(z < 1) {
		b.setMax(center.z, 2);
	}
	else {
		b.setMin(center.z, 2);
	}
	
	if(y < 1) {
		b.setMax(center.y, 1);
	}
	else {
		b.setMin(center.y, 1);
	}
	
	if(x < 1) {
		b.setMax(center.x, 0);
	}
	else {
		b.setMin(center.x, 0);
	}
}

template<typename CellType, typename ValueType, int MaxLevel>
Vector3F AdaptiveGrid3<CellType, ValueType, MaxLevel>::cellCenter(const Coord4 & c) const
{
	BoundingBox b;
	getCellBBox(b, c);
	
	return b.center();
}

template<typename CellType, typename ValueType, int MaxLevel>
Vector3F AdaptiveGrid3<CellType, ValueType, MaxLevel>::cellChildCenter(const Coord4 & c,
						const int & i) const
{
	BoundingBox b;
	getCellChildBox(b, i, c);
	return b.center();
}

template<typename CellType, typename ValueType, int MaxLevel>
Coord4 AdaptiveGrid3<CellType, ValueType, MaxLevel>::parentCoord(const Coord4 & c) const
{
	const Vector3F q = cellCenter(c);
	return cellCoordAtLevel(q, c.w - 1);
}

template<typename CellType, typename ValueType, int MaxLevel>
Coord4 AdaptiveGrid3<CellType, ValueType, MaxLevel>::neighborCoord(const Coord4 & c, const int & i) const
{
	int g = m_levelCoord[c.w];
	Coord4 nc;
	nc.x = c.x + g * gdt::TwentySixNeighborOffset[i][0];
	nc.y = c.y + g * gdt::TwentySixNeighborOffset[i][1];
	nc.z = c.z + g * gdt::TwentySixNeighborOffset[i][2];
	nc.w = c.w;
	return nc;
}

template<typename CellType, typename ValueType, int MaxLevel>
CellType * AdaptiveGrid3<CellType, ValueType, MaxLevel>::findNeighborCell(const Coord4 & c,
						const int & i)
{
	Coord4 nc = neighborCoord(c, i);
	return findCell(nc);
}

template<typename CellType, typename ValueType, int MaxLevel>
bool AdaptiveGrid3<CellType, ValueType, MaxLevel>::isCellFaceOnBoundray(const Coord4 & cellCoord, const int & i)
{
	int g = m_levelCoord[cellCoord.w];
	Coord4 nc;
	nc.x = cellCoord.x + g * gdt::TwentySixNeighborOffset[i][0];
	nc.y = cellCoord.y + g * gdt::TwentySixNeighborOffset[i][1];
	nc.z = cellCoord.z + g * gdt::TwentySixNeighborOffset[i][2];
	nc.w = cellCoord.w;
	Vector3F q = cellCenter(nc);
	nc = cellCoordAtLevel(q, 0);
	if(findCell(nc) )
		return false;
		
	return true;
}

}

}