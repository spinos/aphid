/*
 *  AdaptiveBccGrid.cpp
 *  ttg
 *
 *  Created by jian zhang on 7/21/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaptiveBccGrid3.h"

namespace aphid {

namespace ttg {

AdaptiveBccGrid3::AdaptiveBccGrid3()
{}

AdaptiveBccGrid3::~AdaptiveBccGrid3()
{}

BccCell3 * AdaptiveBccGrid3::addCell(const sdb::Coord4 & cellCoord)
{
#if 0
	std::cout<<"\n AdaptiveBccGrid3::addCell";
#endif
	BccCell3 * c = AdaptiveGrid10T::addCell(cellCoord);
	c->insertRed( cellCenter(cellCoord) );
	
	return c;
}

void AdaptiveBccGrid3::subdivideCells(const std::vector<sdb::Coord4 > & divided)
{
	std::vector<sdb::Coord4 >::const_iterator it = divided.begin();
	for(;it!=divided.end();++it) {
		subdivideCell(*it);
	}
}

void AdaptiveBccGrid3::build()
{
#if 0
	std::cout<<"\n AdaptiveBccGrid3::build";
#endif
	begin();
	while(!end() ) {
		const sdb::Coord4 k = key();
		BccCell3 * cell = value();
		cell->insertFaceOnBoundary(k, this);
		cell->insertBlue(k, this);
		if(cell->hasChild() ) {
			cell->insertYellow(k, this);
			cell->insertCyan(k, this);
		}
		
		next();
	}
	countNodes();
}

void AdaptiveBccGrid3::subdivideToLevel(const BoundingBox & bx, 
						const int & level,
						std::vector<sdb::Coord4 > * divided)
{
#if 0
	std::cout<<"\n AdaptiveBccGrid3::subdivideToLevel"<<bx<<" "<<level;
#endif
	const int s = level0CoordStride();
	const sdb::Coord4 lc = cellCoordAtLevel(bx.getMin(), 0);
	const sdb::Coord4 hc = cellCoordAtLevel(bx.getMax(), 0);
	const int dimx = (hc.x - lc.x) / s + 1;
	const int dimy = (hc.y - lc.y) / s + 1;
	const int dimz = (hc.z - lc.z) / s + 1;
	const float fh = finestCellSize();
	
	//std::cout<<"\n level0 cell stride "<<s
	//		<<"\n grid dim "<<dimx<<" x "<<dimy<<" x "<<dimz;
	
	const Vector3F ori(fh * (lc.x + s/2),
						fh * (lc.y + s/2),
						fh * (lc.z + s/2));
				
	int i, j, k;
	sdb::Coord4 sc;
	sc.w = 0;
	for(k=0; k<dimz;++k) {
		sc.z = lc.z + s * k;
			for(j=0; j<dimy;++j) {
			sc.y = lc.y + s * j;
			for(i=0; i<dimx;++i) {
				
				sc.x = lc.x + s * i;
				BccCell3 * cell = findCell(sc);
				if(cell) {
					if(level > 0) {
						subdivideCellToLevel(cell, sc, bx, level, divided);
					}
				} else { 
					if(level < 1) {
						addCell(sc);
					}
				}
				
			}
		}
	}
	
}

void AdaptiveBccGrid3::subdivideCellToLevel(BccCell3 * cell,
						const sdb::Coord4 & cellCoord,
						const BoundingBox & bx, 
						const int & level,
						std::vector<sdb::Coord4 > * divided)
{
	int i;
	if(!cell->hasChild() ) {
/// add 8 children
		for(i=0; i< 8; ++i) { 
			subdivide(cellCoord, i);
		}
	}
	
	if(cellCoord.w > 0) {
		if(divided) {
			divided->push_back(cellCoord);
		}
	}
	
	if(cellCoord.w + 1 == level) {
		return;
	}
	
	BoundingBox childBx;
/// subdidive each child if intersect
	for(i=0; i< 8; ++i) {
		getCellChildBox(childBx, i, cellCoord);
		if(childBx.intersect(bx) ) {
			
			const sdb::Coord4 cc = childCoord(cellCoord, i);
			BccCell3 * childCell = findCell(cc);
			subdivideCellToLevel(childCell, cc, bx, level, divided);
		}
	}
}

bool AdaptiveBccGrid3::subdivideCell(const sdb::Coord4 & cellCoord,
						std::vector<sdb::Coord4 > * divided)
{
	BccCell3 * cell = findCell(cellCoord);
	if(!cell) {
		std::cout<<"\n [ERROR] cannot find cell to subdivide "<<cellCoord;
		return false;
	}
	
	if(cell->hasChild() ) {
		return false;
	}
	
/// add 8 children
	for(int i=0; i< 8; ++i) { 
		subdivide(cellCoord, i);
		
	}
	
	if(divided) {
		divided->push_back(cellCoord);
	}
	
	return true;
}

void AdaptiveBccGrid3::enforceBoundary(std::vector<sdb::Coord4 > & ks)
{
#if 1
	std::cout<<"\n AdaptiveBccGrid3::enforceBoundar "<<ks.size();
#endif
	while(ks.size() > 0) {
/// first one
		const sdb::Coord4 c = ks[0];
		
/// per face
		for(int i=0; i< 6;++i) {
			const sdb::Coord4 nei = neighborCoord(c, i);
			const sdb::Coord4 par = parentCoord(nei);
			if(findCell(par) ) {
				if(!findCell(nei) ) {
					if(subdivideCell(par ) )
/// last one
						ks.push_back(par);
				}
			}
		}
		
/// per edge
		for(int i=14; i< 26;++i) {
			const sdb::Coord4 nei = neighborCoord(c, i);
			const sdb::Coord4 par = parentCoord(nei);
			if(findCell(par) ) {
				if(!findCell(nei) ) {
					if(subdivideCell(par ) )
/// last one
						ks.push_back(par);
				}
			}
		}
		
/// rm first one
		ks.erase(ks.begin() );
	}
}

}
}
