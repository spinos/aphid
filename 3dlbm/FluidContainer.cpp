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
	for(unsigned k = 0; k < fGridZ; k++) {
	    for(unsigned j = 0; j < fGridY; j++) {
	        for(unsigned i = 0; i < fGridX; i++) {
	            fDensity[k * (fGridX * fGridY) + j * fGridX + i] = 0.f;
	        }
	    }
	}
	fDrawer = new ShapeDrawer();

}

void FluidContainer::killPhysics()
{
	delete[] fDensity;
}

void FluidContainer::renderWorld()
{
	fDrawer->box(fGridSize * fGridX, fGridSize * fGridY, fGridSize * fGridZ);
	fDrawer->setGrey(1.f);
	for(unsigned k = 0; k < fGridZ; k++) {
	    for(unsigned j = 0; j < fGridY; j++) {
	        for(unsigned i = 0; i < fGridX; i++) {
	            const float d = fDensity[k * (fGridX * fGridY) + j * fGridX + i];
	            if(d > 0.001f)
	                fDrawer->solidCube(i * fGridSize, j * fGridSize, k * fGridSize, fGridSize);
	        }
	    }
	}
}

void FluidContainer::simulate()
{
	//btScalar dt = (btScalar)_clock.getTimeMicroseconds();
	//_clock.reset();
	//_dynamicsWorld->stepSimulation(dt / 1000000.f, 10);
}

void FluidContainer::addSource(const Vector3F & pos, const Vector3F & dir)
{
    int gx = pos.x / fGridSize;
    if( gx < 0 || gx >= fGridX )
        return;
    
    int gy = pos.y / fGridSize;
    if( gy < 0 || gy >= fGridY )
        return;
    
    int gz = pos.z / fGridSize;
    if( gz < 0 || gz >= fGridZ )
        return;
    
    fDensity[gz * (fGridX * fGridY) + gy * fGridX + gx] = 1.f;
}

