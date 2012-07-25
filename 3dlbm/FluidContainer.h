/*
 *  FluidContainer.h
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "shapeDrawer.h"
#include <Vector3F.h>

class FluidContainer {
public:
	FluidContainer();
	virtual ~FluidContainer();
	
	void initPhysics();
	void killPhysics();
	void renderWorld();
	void simulate();
	void addSource(const Vector3F & pos, const Vector3F & dir);
	
protected:
	ShapeDrawer* fDrawer;
	unsigned fGridX, fGridY, fGridZ;
	float fGridSize;
	float *fDensity;
};