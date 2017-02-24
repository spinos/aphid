/*
 *  AdaptiveBccGrid3.h
 *  ttg
 *
 *  Created by jian zhang on 7/21/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_ADAPTIVE_BCC_GRID_3_H
#define APH_TTG_ADAPTIVE_BCC_GRID_3_H

#include <ttg/BccCell3.h>
#include <BDistanceFunction.h>

namespace aphid {

namespace ttg {

typedef sdb::AdaptiveGrid3<BccCell3, BccNode3, 10 > AdaptiveGrid10T;

class AdaptiveBccGrid3 : public AdaptiveGrid10T {

public:
	AdaptiveBccGrid3();
	virtual ~AdaptiveBccGrid3();
/// subdivide intersected cells to 8 sub-cells
/// enforce cell boundray           
    template<typename Tf>
	void subdivideToLevel(Tf & fintersect,
						sdb::AdaptiveGridDivideProfle & prof)
	{
#if 0
		std::cout<<"\n AdaptiveBccGrid3::subdivide ";
#endif
		std::vector<sdb::Coord4 > divided;

		BoundingBox cb;
		int level = prof.minLevel();
		while(level < prof.maxLevel() ) {
			std::vector<sdb::Coord4> dirty;
			begin();
			while(!end() ) {
				if(key().w == level) {
					getCellBBox(cb, key() );
					
					if(limitBox().intersect(cb) ) {
						if(fintersect.intersect(cb) ) {
							dirty.push_back(key() );
						}
					}
				}
				
				next();
			}
#if 0			
			std::cout<<"\n subdiv level "<<level<<" divided "<<divided.size();
#endif
			std::vector<sdb::Coord4>::const_iterator it = dirty.begin();
			for(;it!=dirty.end();++it) {
                getCellBBox(cb, *it);
				cb.expand(-1e-4f);
				BccCell3 * cell = findCell(*it);
                subdivideCellToLevel(cell, *it, cb, level+1, &divided);
			}
			level++;
		}
		
        enforceBoundary(divided);
        storeCellNeighbors();
		storeCellChildren();
	}
	
/// create node in each cell
	void build();
	
/// find level cell intersect distance function
	template<typename Tf>
	void markCellIntersectDomainAtLevel(Tf * d, 
						const int & level,
						std::vector<sdb::Coord4 > & divided)
	{
		BoundingBox cb;
		begin();
		while(!end() ) {
			const sdb::Coord4 k = key();
			
			if(k.w == level) {
				getCellBBox(cb, k);
				
				if(d-> template broadphase <BoundingBox>(&cb ) )
					divided.push_back(k);
			}
			
			next();
		}
	}

	void subdivideCells(const std::vector<sdb::Coord4 > & divided);
/// add 8 child cells
/// to level + 1
	bool subdivideCell(const sdb::Coord4 & cellCoord,
						std::vector<sdb::Coord4 > * divided = 0);
/// subdivide recursively to level
	void subdivideToLevel(const BoundingBox & bx, 
						const int & level,
						std::vector<sdb::Coord4 > * divided = 0);	
						
protected:
	void subdivideCellToLevel(BccCell3 * cell,
						const sdb::Coord4 & cellCoord,
						const BoundingBox & bx, 
						const int & level,
						std::vector<sdb::Coord4 > * divided);					
	BccCell3 * addCell(const sdb::Coord4 & cellCoord);
	
private:
	
/// for each cell divied, must have same level neighbor cell on six faces and twelve edges
/// level change cross face or edge <= 1
	void enforceBoundary(std::vector<aphid::sdb::Coord4 > & ks);

};

}
}
#endif
