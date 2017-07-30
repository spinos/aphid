/*
 *  WindForce.cpp
 *  
 *
 *  Created by jian zhang on 7/31/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "WindForce.h"
#include <math/Vector3F.h>

namespace aphid {

namespace pbd {

float WindForce::Cdrag = 0.75f;
float WindForce::Clift = 0.05f;

WindForce::WindForce()
{}

Vector3F WindForce::ComputeDragAndLift(const Vector3F& vair, const Vector3F& nml)
{
	float vdn = vair.normal().dot(nml);
	Vector3F Fdl = vair * ((Cdrag - Clift) );
	Fdl += nml * (Clift * vair.length() * vdn);
	return Fdl;
}

}
}
