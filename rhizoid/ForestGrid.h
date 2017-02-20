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
	int m_numVisibleSamples;
	
public:
	ForestGrid();
	virtual ~ForestGrid();
	
	const int & numActiveCells() const;
	const int & numActiveSamples() const;
	const int & numVisibleSamples() const;
	
/// T as intersect type, Tc as closest to point type
/// Tf as select filter type
	template<typename T, typename Tc, typename Tf, typename Tshape>
	void selectCells(T & ground,
					Tc & closestGround,
					Tf & selFilter,
					const Tshape & selShape);
					
/// for all active cells, reshuffle samples noise
	void reshuffleSamples(const int & level);
					
	void activeCellBegin();
	void activeCellNext();
	const bool activeCellEnd() const;
	ForestCell * activeCellValue();
	const sdb::Coord3 & activeCellKey() const;
	
	void deselectCells();
	int countActiveSamples();
	void clearSamplles();
	
	int countPlants();
	void clearPlants();
	
	template<typename Tf>
	void processFilter(Tf & selFilter);
	
};

template<typename T, typename Tc, typename Tf, typename Tshape>
void ForestGrid::selectCells(T & ground,
					Tc & closestGround,
					Tf & selFilter,
					const Tshape & selShape)
{
	if(selFilter.isReplacing() ) {
		deselectCells();
	}
	
	BoundingBox shapeBx = selShape.calculateBBox();
	shapeBx.shrinkBy(ground.getBBox() );
	
	const Vector3F plow = shapeBx.lowCorner();
	const Vector3F phigh = shapeBx.highCorner();			
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
				
				cell-> template selectSamples<Tf, Tshape>(selFilter, selShape);
								
				if(cell->numActiveSamples() > 0) {
					m_activeCells.insert(sc, cell);
					cell-> template processFilter<Tf>(selFilter);
				} else {
					m_activeCells.remove(sc);
				}
				
			}
		}
	}
	m_numActiveCells = m_activeCells.size();
}

template<typename Tf>
void ForestGrid::processFilter(Tf & selFilter)
{
	const int & level = selFilter.maxSampleLevel();
	begin();
	while(!end() ) {
		ForestCell * cell = value();
		if(cell->hasSamples(level) ) {
			cell->processFilter(selFilter);
		}
		
		next();
	}
}

}
#endif