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
	BccCell3 * c = sdb::AdaptiveGrid3<BccCell3, BccNode >::addCell(pref, level);
	
	const sdb::Coord4 k = cellCoordAtLevel(pref, level);
	
	c->insertRed( cellCenter(k) );
	return c;
}

BccCell3 * AdaptiveBccGrid3::subdivide(const sdb::Coord4 & cellCoord, const int & i)
{
	BccCell3 * c = sdb::AdaptiveGrid3<BccCell3, BccNode >::subdivide(cellCoord, i);
	
	c->insertRed( cellChildCenter(cellCoord, i) );
	return c;
}

void AdaptiveBccGrid3::build()
{
	begin();
	while(!end() ) {
		const sdb::Coord4 k = key();
		BccCell3 * cell = value();
		if(k.w == 0) {
			cell->insertBlue(k, this);
			if(cell->hasChild() ) {}
		}
		else {
			
		}
		next();
	}
}

}