/*
 *  VolumeResponse.h
 *
 *  continuum model to solve particle interactions
 *
 *  Created by jian zhang on 1/21/18.
 *  Copyright 2018 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_LBM_VOLUME_RESPONSE_H
#define APH_LBM_VOLUME_RESPONSE_H

#include "LatticeManager.h"

namespace aphid {

namespace lbm {

class VolumeResponse : public LatticeManager {

public:

	VolumeResponse();
/// put particles into grid
/// use particle velocities as initial condition
/// one simulation step
/// modify particle velocities by fluid velocity field
	void solveParticles(float* vel, const float* pos, const int& np);
	
protected:

};

}

}

#endif
