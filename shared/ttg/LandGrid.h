/*
 *  LandGrid.h
 *  
 *  array of land blocks
 *
 *  Created by jian zhang on 3/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_LAND_GRID_H
#define APH_TTG_LAND_GRID_H

#include <sdb/WorldGrid3.h>
#include <LandBlock.h>

namespace aphid {

namespace ttg {

class LandGrid : public sdb::WorldGrid3<LandBlock> {

public:
	LandGrid();
	virtual ~LandGrid();
	
};

}

}
#endif