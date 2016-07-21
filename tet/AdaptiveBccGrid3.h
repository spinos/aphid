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

namespace ttg {

class AdaptiveBccGrid3 : public aphid::sdb::AdaptiveGrid3<BccCell3, BccNode > {

public:
	AdaptiveBccGrid3();
	virtual ~AdaptiveBccGrid3();
	
	BccCell3 * addCell(const aphid::Vector3F & pref, const int & level = 0);
/// add 8 child cells
	void subdivideCell(const aphid::sdb::Coord4 & cellCoord);
/// node to each cell
	void build();
	
private:

};

}