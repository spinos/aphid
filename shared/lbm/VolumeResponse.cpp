/*
 *  VolumeResponse.cpp
 *  
 *
 *  Created by jian zhang on 1/21/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#include "VolumeResponse.h"

namespace aphid {

namespace lbm {

VolumeResponse::VolumeResponse()
{}

void VolumeResponse::solveParticles(float* vel, const float* pos, const int& np)
{
	resetLattice();
	injectParticles(pos, vel, np);
	finishInjectingParticles();
	initialCondition();
	simulationStep();
	modifyParticleVelocities(vel, pos, np);
}

}

}
