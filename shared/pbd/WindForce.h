/*
 *  WindForce.h
 *  
 *
 *  Created by jian zhang on 7/31/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_PBD_WIND_FORCE_H
#define APH_PBD_WIND_FORCE_H

namespace aphid {

class Vector3F;

namespace pbd {

class WindForce {

public:
	WindForce();
	
	static float Cdrag;
	static float Clift;
/// (Cd - Cl) v + Cl|v|(v.n) n
/// vair is relative wind velocity
/// nml is geometry normal
	static Vector3F ComputeDragAndLift(const Vector3F& vair, const Vector3F& nml);

};

}

}
#endif
