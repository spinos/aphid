/*
 *  AccInterior.h
 *  knitfabric
 *
 *  Created by jian zhang on 5/30/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Vector3F.h>
namespace aphid {

class AccInterior {
public:
	AccInterior() {}

	Vector3F computePosition() const;
	Vector3F computeNormal() const;
	
	Vector3F _cornerPositions[4];
	Vector3F _cornerNormals[4];
	float _valence;
private:

};

}