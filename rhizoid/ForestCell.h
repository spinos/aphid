/*
 *  ForestCell.h
 *  proxyPaint
 *
 *  Created by jian zhang on 1/19/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_FOREST_CELL_H
#define APH_FOREST_CELL_H

#include <sdb/Array.h>

namespace aphid {

class Plant;

class ForestCell : public sdb::Array<sdb::Coord2, Plant> {

public:
	ForestCell(Entity * parent = NULL);
	virtual ~ForestCell();
	
protected:

private:
};

}
#endif
