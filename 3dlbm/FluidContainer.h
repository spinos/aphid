/*
 *  FluidContainer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "shapeDrawer.h"


class FluidContainer {
public:
	FluidContainer();
	virtual ~FluidContainer();
	
	void initPhysics();
	void killPhysics();
	void renderWorld();
	void simulate();
	
protected:
	ShapeDrawer* fDrawer;
	unsigned fGridX, fGridY, fGridZ;
	float fGridSize;
	float *fDensity;
};