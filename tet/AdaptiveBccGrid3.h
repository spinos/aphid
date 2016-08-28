/*
 *  AdaptiveBccGrid3.h
 *  foo
 *
 *  Created by jian zhang on 7/21/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BccCell3.h>
#include <BDistanceFunction.h>

namespace ttg {

typedef aphid::sdb::AdaptiveGrid3<BccCell3, BccNode3, 10 > AdaptiveGrid10T;

class AdaptiveBccGrid3 : public AdaptiveGrid10T {

public:
	AdaptiveBccGrid3();
	virtual ~AdaptiveBccGrid3();
	
/// add 8 child cells
	bool subdivideCell(const aphid::sdb::Coord4 & cellCoord);
	
/// create node in each cell
/// reuse distance value in field
	void build(aphid::ADistanceField * fld);
/// find all level0 cell intersect box
/// subdivide recursively to level
	void subdivideToLevel(const aphid::BoundingBox & bx, 
						const int & level,
						std::vector<aphid::sdb::Coord4 > * divided = NULL);
	
	void subdivideCellToLevel(BccCell3 * cell,
						const aphid::sdb::Coord4 & cellCoord,
						const aphid::BoundingBox & bx, 
						const int & level,
						std::vector<aphid::sdb::Coord4 > * divided);
	
/// find level cell intersect distance function
	template<typename Tf>
	void markCellIntersectDomainAtLevel(Tf * d, 
						const int & level,
						std::vector<aphid::sdb::Coord4 > & divided)
	{
		aphid::BoundingBox cb;
		begin();
		while(!end() ) {
			const aphid::sdb::Coord4 k = key();
			
			if(k.w == level) {
				getCellBBox(cb, k);
				
				if(d-> template broadphase <aphid::BoundingBox>(&cb ) )
					divided.push_back(k);
			}
			
			next();
		}
	}
	
	void subdivideCells(const std::vector<aphid::sdb::Coord4 > & divided);
						
private:
	BccCell3 * addCell(const aphid::sdb::Coord4 & cellCoord);
	
	
	
};

}