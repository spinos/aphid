/*
 *  DistanceFieldTest.h
 *  foo
 *
 *  Created by jian zhang on 7/14/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <AGraph.h>
#include "Scene.h"
#include "GridMaker.h"

namespace ttg {

class DistanceFieldTest : public Scene {

	GridMaker m_gridmk;
	
public:
	DistanceFieldTest();
	virtual ~DistanceFieldTest();
	
	virtual const char * titleStr() const;
	virtual bool init();
	virtual void draw(aphid::GeoDrawer * dr);
	
};

}