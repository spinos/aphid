/*
 *  FluidContainer.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/13/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "FluidContainer.h"

FluidContainer::FluidContainer() {}
FluidContainer::~FluidContainer() {}

void FluidContainer::initPhysics()
{
	fGridX = fGridY = fGridZ = 32;
	fGridSize = 1.f;
	fDensity = new float[fGridX * fGridY * fGridZ];
	fDrawer = new ShapeDrawer();

}

void FluidContainer::killPhysics()
{
	delete[] fDensity;
}

void FluidContainer::renderWorld()
{
	fDrawer->box(fGridSize * fGridX, fGridSize * fGridY, fGridSize * fGridZ);
	fDrawer->solidCube(0, 0, 0, 1.f);
}

void FluidContainer::simulate()
{
	//btScalar dt = (btScalar)_clock.getTimeMicroseconds();
	//_clock.reset();
	//_dynamicsWorld->stepSimulation(dt / 1000000.f, 10);
}
