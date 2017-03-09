/*
 *  LandBlock.h
 *  
 *  a single piece of land
 *
 *  Created by jian zhang on 3/9/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_TTG_LAND_BLOCK_H
#define APH_TTG_LAND_BLOCK_H

#include <sdb/Entity.h>
#include <ttg/GenericTetraGrid.h>
#include <ttg/AdaptiveBccGrid3.h>

namespace aphid {

namespace ttg {

class LandBlock : public sdb::Entity {

public:
typedef	ttg::GenericTetraGrid<float > TetGridTyp;

private:
	TetGridTyp m_tetg;
	AdaptiveBccGrid3 m_bccg;
	
public:
	LandBlock(sdb::Entity * parent = NULL);
	virtual ~LandBlock();
	
	const TetGridTyp * grid() const;
	
protected:

private:	
};

}

}
#endif