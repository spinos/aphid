/*
 *  GridSelection.h
 *  
 *	T as grid type, Tc as cell type
 *
 *  Created by jian zhang on 2/16/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_SDB_GRID_SELECTION_H
#define APH_SDB_GRID_SELECTION_H

#include <vector>

namespace aphid {

namespace sdb {

template<typename T, typename Tc>
class GridSelection {

	T * m_grid;
	int m_maxSelectLevel;
	
public:
	struct CellInd {
		Coord4 key;
		Tc * value;
	};
	
private:

typedef std::vector<CellInd> CellIndArrayT;
	CellIndArrayT m_activeCells;
	
public:
	GridSelection(T * grid);
	
	void setMaxSelectLevel(int x);
	
	bool select(const BoundingBox & selBx);
				
	bool select(const Vector3F & center,
				const float & radius);
				
	const int numActiveCells() const;
	Tc * activeCell(const int & i) const;
	const Coord4 & activeCellCoord(const int & i) const;

protected:
	
private:
	void selectInCell(const BoundingBox & selBx,
						const CellInd & ind);
	
};

template<typename T, typename Tc>
GridSelection<T, Tc>::GridSelection(T * grid)
{ m_grid = grid; }

template<typename T, typename Tc>
void GridSelection<T, Tc>::setMaxSelectLevel(int x)
{ m_maxSelectLevel = x; }

template<typename T, typename Tc>
bool GridSelection<T, Tc>::select(const BoundingBox & selBx)
{						
	BoundingBox cb;
	int level = 0;
/// select level 0
	CellIndArrayT dirtyParent;
	CellInd ci;
	
	m_grid->begin();
	while(!m_grid->end() ) {
		ci.key = m_grid->key();
		if(ci.key.w == level) {
			m_grid->getCellBBox(cb, ci.key );
			
			if(selBx.intersect(cb) ) {
				ci.value = m_grid->value();
				dirtyParent.push_back(ci);
			}
			
		}
		
		if(ci.key.w > level ) {
			break;
		}
		
		m_grid->next();
	}
	
	if(dirtyParent.size() < 1) {
		return false;
	}
	
	while(level < m_maxSelectLevel) {
		m_activeCells.clear();
		
		const int n = dirtyParent.size();
		for(int i=0;i<n;++i) {
			selectInCell(selBx, dirtyParent[i]);
		}
		
		if(m_activeCells.size() < 1) {
			return false;
		}
		
		level++;
		
		if(level < m_maxSelectLevel) {
/// swap
			dirtyParent.clear();
			const int nc = m_activeCells.size();
			for(int i=0;i<nc;++i) {
				dirtyParent.push_back(m_activeCells[i]);
			}
		
		}
		
	}
	
	return true;
}

template<typename T, typename Tc>
bool GridSelection<T, Tc>::select(const Vector3F & center,
				const float & radius)
{
	const BoundingBox selBx(center.x - radius, 
						center.y - radius,
						center.z - radius,
						center.x + radius, 
						center.y + radius,
						center.z + radius);
	return select(selBx);
}

template<typename T, typename Tc>
const int GridSelection<T, Tc>::numActiveCells() const
{ return m_activeCells.size(); }

template<typename T, typename Tc>
Tc * GridSelection<T, Tc>::activeCell(const int & i) const
{ return m_activeCells[i].value; }

template<typename T, typename Tc>
const Coord4 & GridSelection<T, Tc>::activeCellCoord(const int & i) const
{ return m_activeCells[i].key; }

template<typename T, typename Tc>
void GridSelection<T, Tc>::selectInCell(const BoundingBox & selBx,
						const CellInd & ind)
{
	if(!ind.value->hasChild() ) {
		return;
	}
	
	BoundingBox cb;
	CellInd ci;
	for(int i=0; i< 8; ++i) { 
		ci.key = m_grid->childCoord(ind.key, i);
		Tc * childCell = m_grid->findCell(ci.key);
		if(childCell) {
			m_grid->getCellChildBox(cb, i, ind.key );
			if(selBx.intersect(cb) ) {
				ci.value = childCell;
				m_activeCells.push_back(ci);
			}
		}
		
	}
}

}

}
#endif