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
    template<typename Tf, typename Tclosest>
	void subdivideToLevel(Tf & fintersect, Tclosest & fclosest,
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
					
					bool stat = limitBox().intersect(cb);
					if(stat) {
						stat = fintersect.intersect(cb);
					}
					
					if(stat) {
						if(forceSubdivide(level, limitBox(), cb) ) {
							stat = true;
						} else {
							if(prof.minNormalDistribute() < 1.f) {
								fclosest.select(cb);
								stat = fclosest.normalDistributeBelow(prof.minNormalDistribute() );
							} else {
								stat = true;
							}
						}
					}
					
					if(stat) {
						dirty.push_back(key() );
					}
				}
				
				next();
			}
			
			if(dirty.size() < 1) {
				break;
			}
#if 0			
			std::cout<<"\n subdiv level "<<level<<" divided "<<divided.size();
#endif
			std::vector<sdb::Coord4>::const_iterator it = dirty.begin();
			for(;it!=dirty.end();++it) {
                getCellBBox(cb, *it);
				cb.expand(-1e-3f);
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
			
	template<typename T>
	void buildHexagonGrid(T * hexag, int level);
	
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

/// low level or on boundary
	bool forceSubdivide(int level, 
					const BoundingBox & limitBx,
					const BoundingBox & bx) const;
	
};

template<typename T>
void AdaptiveBccGrid3::buildHexagonGrid(T * hexag, int level)
{
#if 0
	std::cout<<"\n AdaptiveBccGrid3::buildHexagonGrid "<<level;
	std::cout.flush();
#endif

typedef sdb::Couple<Vector3F, int> PosIndTyp;
	sdb::Array<int, PosIndTyp > pnts;
	std::vector<int> inds;
	
	int nc = 0;
	begin();
	while(!end() ) {
		const sdb::Coord4 k = key();
		if(k.w == level) {
			BccCell3 * cell = value();
			cell->getBlueVertices(pnts, inds, k, this);
			nc++;
		
		} else if(k.w > level) {
			break;
		}
		
		next();
	}
	
	int np = 0;
	pnts.begin();
	while(!pnts.end() ) {
		int * ind = pnts.value()->t2;
		*ind = np;
		np++;
		pnts.next();
	}

#if 0
	std::cout<<"\n np "<<np<<" nc "<<nc;
	std::cout.flush();
#endif

	hexag->create(np, nc);
	
	pnts.begin();
	while(!pnts.end() ) {
		const PosIndTyp * pind = pnts.value();
		
		hexag->setPos(*pind->t1, *pind->t2);
		
		pnts.next();
	}
	
	int hexav[8];
	for(int i=0;i<nc;++i) {
		const int cf = i<<3;
		for(int j=0;j<8;++j) {
			hexav[j] = *(pnts.find(inds[cf+j])->t2);
		}
		
		hexag->setCell(hexav, i);
	}
	
	pnts.clear();
	
}

}
}
#endif
