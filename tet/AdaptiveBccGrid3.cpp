/*
 *  AdaptiveBccGrid.cpp
 *  foo
 *
 *  Created by jian zhang on 7/21/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "AdaptiveBccGrid3.h"

using namespace aphid;

namespace ttg {

AdaptiveBccGrid3::AdaptiveBccGrid3()
{}

AdaptiveBccGrid3::~AdaptiveBccGrid3()
{}

BccCell3 * AdaptiveBccGrid3::addCell(const aphid::sdb::Coord4 & cellCoord)
{
	BccCell3 * c = AdaptiveGrid10T::addCell(cellCoord);
	
	c->insertRed( cellCenter(cellCoord) );
	
	return c;
}

bool AdaptiveBccGrid3::subdivideCell(const sdb::Coord4 & cellCoord)
{
	BccCell3 * cell = findCell(cellCoord);
	if(!cell) {
		std::cout<<"\n [ERROR] cannot find cell to subdivide "<<cellCoord;
		return false;
	}
	
	if(cell->hasChild() ) {
		//std::cout<<"\n [WARNING] cell already divided "<<cellCoord;
		return false;
	}
	
	for(int i=0; i< 8; ++i) { 
		BccCell3 * c = subdivide(cellCoord, i);
	
		c->insertRed( childCenter(cellCoord, i) );
	}
	return true;
}

void AdaptiveBccGrid3::subdivideCells(const std::vector<sdb::Coord4 > & divided)
{
	std::vector<sdb::Coord4 >::const_iterator it = divided.begin();
	for(;it!=divided.end();++it)
		subdivideCell(*it);
}

void AdaptiveBccGrid3::build(ADistanceField * fld)
{
	begin();
	while(!end() ) {
		const sdb::Coord4 k = key();
		BccCell3 * cell = value();
		cell->insertFaceOnBoundary(k, this);
		cell->insertBlue(k, this, fld);
		if(cell->hasChild() ) {
			cell->insertYellow(k, this, fld);
			cell->insertCyan(k, this, fld);
		}
		
		next();
	}
	countNodes();
	
}

void AdaptiveBccGrid3::subdivideToLevel(const BoundingBox & bx, 
						const int & level,
						std::vector<sdb::Coord4 > * divided)
{
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
					if(level > 0)
						subdivideCellToLevel(cell, sc, bx, level, divided);
				}
				else { 
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
			BccCell3 * c = subdivide(cellCoord, i);
	
			c->insertRed( childCenter(cellCoord, i) );
		}
		
		if(cellCoord.w > 0) {
			if(divided)
				divided->push_back(cellCoord);
		}
	}
	
	if(cellCoord.w + 1 == level)
		return;
	
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

}