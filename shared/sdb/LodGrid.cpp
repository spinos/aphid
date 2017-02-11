/*
 *  LodGrid.cpp
 *  
 *
 *  Created by jian zhang on 2/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "LodGrid.h"
#include <math/miscfuncs.h>

namespace aphid {

namespace sdb {

LodNode::LodNode()
{}

LodNode::~LodNode()
{}

LodCell::LodCell(Entity * parent) :
sdb::Array<int, LodNode >(parent)
{}

LodCell::~LodCell()
{}

void LodCell::clear()
{ 
	TParent::clear();
}

void LodCell::countNodesInCell(int & it)
{
	begin();
	while(!end() ) {
		value()->index = it;
		it++;

		next();
	}
}

void LodCell::dumpNodesInCell(LodNode * dst)
{
	begin();
	while(!end() ) {
		LodNode * a = value();
		dst[a->index] = *a;
		
		next();
	}
}

LodGrid::LodGrid()
{}

LodGrid::~LodGrid()
{}

void LodGrid::fillBox(const BoundingBox & b,
				const float & h)
{
	clear();
	setLevel0CellSize(h);
	
	const int s = level0CoordStride();
	const sdb::Coord4 lc = cellCoordAtLevel(b.getMin(), 0);
	const sdb::Coord4 hc = cellCoordAtLevel(b.getMax(), 0);
	const int dimx = (hc.x - lc.x) / s + 1;
	const int dimy = (hc.y - lc.y) / s + 1;
	const int dimz = (hc.z - lc.z) / s + 1;
	const float fh = finestCellSize();
	
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
				LodCell * cell = findCell(sc);
				if(!cell) { 
					addCell(sc);
				}
				
			}
		}
	}
	
	calculateBBox();
}

void LodGrid::clear()
{
	TParent::clear(); 
}

int LodGrid::countLevelNodes(int level)
{
	int c = 0;
	begin();
	while(!end() ) {
		if(key().w == level) {
			value()->countNodesInCell(c);
		}
		
		if(key().w > level) {
			break;
		}

		next();
	}
	
	return c;
}

void LodGrid::dumpLevelNodes(LodNode * dst, int level)
{
	begin();
	while(!end() ) {
		if(key().w == level) {
			value()->dumpNodesInCell(dst);
		}
		
		if(key().w > level) {
			break;
		}

		next();
	}
}

}

}
