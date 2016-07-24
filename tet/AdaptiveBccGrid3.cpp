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

BccCell3 * AdaptiveBccGrid3::addCell(const Vector3F & pref, const int & level)
{
	BccCell3 * c = AdaptiveGrid5T::addCell(pref, level);
	
	const sdb::Coord4 k = cellCoordAtLevel(pref, level);
	
	c->insertRed( cellCenter(k) );
	
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
		std::cout<<"\n [WARNING] cell already divided "<<cellCoord;
		return false;
	}
	
	for(int i=0; i< 8; ++i) { 
		BccCell3 * c = AdaptiveGrid5T::subdivide(cellCoord, i);
	
		c->insertRed( childCenter(cellCoord, i) );
	}
	return true;
}

void AdaptiveBccGrid3::build()
{
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
}

}