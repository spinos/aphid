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
    
/// reset level 0 w size h then fill box b 
    void fillBox(const BoundingBox & b,
                const float & h);
                
    template<typename Tf>
	void subdivideToLevel(Tf & fintersect,
						int minLevel, int maxLevel)
	{
        std::vector<aphid::sdb::Coord4 > divided;
		BoundingBox cb;
		int level = minLevel;
		while(level < maxLevel) {
			std::vector<sdb::Coord4> dirty;
			begin();
			while(!end() ) {
				if(key().w == level) {
					getCellBBox(cb, key() );
					
					if(fintersect.intersect(cb) ) {
						dirty.push_back(key() );
                    }
				}
				next();
			}
			
			// std::cout<<"\n level"<<level<<" divd "<<dirty.size();
			
			std::vector<sdb::Coord4>::const_iterator it = dirty.begin();
			for(;it!=dirty.end();++it) {
                getCellBBox(cb, *it);
                subdivideToLevel(cb, level+1, &divided);
			}
			level++;
		}
        enforceBoundary(divided);
        divided.clear();
		// storeCellNeighbors();
	}
	
/// add 8 child cells
	bool subdivideCell(const sdb::Coord4 & cellCoord);
	
/// create node in each cell
	void build();
/// find all level0 cell intersect box
/// subdivide recursively to level
	void subdivideToLevel(const BoundingBox & bx, 
						const int & level,
						std::vector<sdb::Coord4 > * divided = NULL);
						
	void subdivideCellToLevel(BccCell3 * cell,
						const sdb::Coord4 & cellCoord,
						const BoundingBox & bx, 
						const int & level,
						std::vector<sdb::Coord4 > * divided);
					
/// to level + 1
	void subdivideCell(const sdb::Coord4 & cellCoord,
						std::vector<sdb::Coord4 > * divided);
	
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
						
private:
	BccCell3 * addCell(const sdb::Coord4 & cellCoord);
/// for each cell divied, must have same level neighbor cell on six faces and twelve edges
/// level change cross face or edge <= 1
	void enforceBoundary(std::vector<aphid::sdb::Coord4 > & ks);

};

}
}
#endif
