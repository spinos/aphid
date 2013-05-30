/*
 *  AccInterior.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 5/30/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "AccInterior.h"

Vector3F AccInterior::computePosition() const
{
	Vector3F res = _cornerPositions[0] * _valence;
	res += _cornerPositions[1] * 2.f;
	res += _cornerPositions[3] * 2.f;
	res += _cornerPositions[2];
	return res / (_valence + 5.f);
}

Vector3F AccInterior::computeNormal() const
{
	Vector3F res = _cornerNormals[0] * _valence;
	res += _cornerNormals[1] * 2.f;
	res += _cornerNormals[3] * 2.f;
	res += _cornerNormals[2];
	return res / (_valence + 5.f);
	res.normalize();
	return res;
}