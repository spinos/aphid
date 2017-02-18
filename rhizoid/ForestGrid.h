/*
 *  ForestGrid.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/18/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_RHIZOID_FOREST_GRID_H
#define APH_RHIZOID_FOREST_GRID_H

#include <sdb/WorldGrid.h>
#include <kd/ClosestToPointEngine.h>
#include "ForestCell.h"

namespace aphid {

class ForestCell;
class Plant;

class ForestGrid : public sdb::WorldGrid<ForestCell, Plant > {

	sdb::Array<sdb::Coord3, ForestCell > m_activeCells;
	int m_numActiveCells;
	int m_numActiveSamples;
	
public:
	ForestGrid();
	virtual ~ForestGrid();
	
	const int & numActiveCells() const;
	const int & numActiveSamples() const;
	
/// T as intersect type, Tc as closest to point type
/// Tf as select filter type
	template<typename T, typename Tc, typename Tf>
	void selectCells(T & ground,
					Tc & closestGround,
					Tf & selFilter);
					
/// in active cells
	template<typename T, typename Tc, typename Tf>
	void rebuildSamples(T & ground,
					Tc & closestGround,
					Tf & selFilter);
					
	void activeCellBegin();
	void activeCellNext();
	const bool activeCellEnd() const;
	ForestCell * activeCellValue();
	const sdb::Coord3 & activeCellKey() const;
	
	void deselectCells();
	int countActiveSamples();
	void clearSamplles();
	
	int countPlants();
	
};

template<typename T, typename Tc, typename Tf>
void ForestGrid::selectCells(T & ground,
					Tc & closestGround,
					Tf & selFilter)
{
	if(selFilter.isReplacing() ) {
		deselectCells();
	}
	
	const Vector3F plow = selFilter.boxLow();
	const Vector3F phigh = selFilter.boxHigh();			
	const sdb::Coord3 lc = gridCoord((const float *)&plow);
	const sdb::Coord3 hc = gridCoord((const float *)&phigh);
	
	const int dimx = (hc.x - lc.x) + 1;
	const int dimy = (hc.y - lc.y) + 1;
	const int dimz = (hc.z - lc.z) + 1;
	int i, j, k;
	sdb::Coord3 sc;
	BoundingBox cbx;
	for(k=0; k<=dimz;++k) {
		sc.z = lc.z + k;
		for(j=0; j<=dimy;++j) {
			sc.y = lc.y + j;
			for(i=0; i<=dimx;++i) {
				sc.x = lc.x + i;
				
				cbx = coordToGridBBox(sc);
				
				if(!ground.intersect(cbx) ) {
					continue;
				}
				
				ForestCell * cell = findCell(sc);
				if(!cell) { 
					cell = insertCell(sc);
				}
				if(!cell->hasSamples(selFilter.maxSampleLevel() ) ) {
					cell->buildSamples(ground, closestGround, selFilter,
								cbx);
				}
								
				if(cell-> template selectSamples<Tf>(selFilter) ) {
					m_activeCells.insert(sc, cell);
				} else {
					m_activeCells.remove(sc);
				}
				
			}
		}
	}
	m_numActiveCells = m_activeCells.size();
}

template<typename T, typename Tc, typename Tf>
void ForestGrid::rebuildSamples(T & ground,
					Tc & closestGround,
					Tf & selFilter)
{
	BoundingBox cbx;
	m_activeCells.begin();
	while(!m_activeCells.end() ) {
		const sdb::Coord3 & k = m_activeCells.key();
		cbx = coordToGridBBox(k);
		ForestCell * cell = m_activeCells.value();
		cell->rebuildSamples(ground, closestGround, selFilter,
					cbx);
		
		m_activeCells.next();
	}
}

}
#endif