/*
 *  AccEdge.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 5/30/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "AccEdge.h"

namespace aphid {

void AccEdge::reset()
{
	_isZeroLength = 0;
	_isBoundary = 0;
}

Vector3F AccEdge::computePosition(int side) const
{
	if(_isZeroLength) return _edgePositions[0];
	
	if(_isBoundary) {
		if(side == 0)
			return _edgePositions[0] * (2.f / 3.f) + _edgePositions[1] * (1.f / 3.f);
		else
			return _edgePositions[1] * (2.f / 3.f) + _edgePositions[0] * (1.f / 3.f);
	}

	Vector3F res;
	
	if(side == 0) {
		res = _edgePositions[0] * 2.f * _valence[0];
		res += _edgePositions[1] * 4.f;
		res += _fringePositions[0] * 2.f;
		res += _fringePositions[1] * 2.f;
		res += _fringePositions[2];
		res += _fringePositions[3];
		res = res / (2.f * _valence[0] + 10.f);
	}
	else {
		res = _edgePositions[1] * 2.f * _valence[1];
		res += _edgePositions[0] * 4.f;
		res += _fringePositions[2] * 2.f;
		res += _fringePositions[3] * 2.f;
		res += _fringePositions[0];
		res += _fringePositions[1];
		res = res / (2.f * _valence[1] + 10.f);
	}
	return res;
}

Vector3F AccEdge::computeNormal(int side) const
{
	if(_isZeroLength) return _edgeNormals[0];
	
	if(_isBoundary) {
		if(side == 0)
			return _edgeNormals[0] * (2.f / 3.f) + _edgeNormals[1] * (1.f / 3.f);
		else
			return _edgeNormals[1] * (2.f / 3.f) + _edgeNormals[0] * (1.f / 3.f);
	}

	Vector3F res;
	
	if(side == 0) {
		res = _edgeNormals[0] * 2.f * _valence[0];
		res += _edgeNormals[1] * 4.f;
		res += _fringeNormals[0] * 2.f;
		res += _fringeNormals[1] * 2.f;
		res += _fringeNormals[2];
		res += _fringeNormals[3];
		res = res / (2.f * _valence[0] + 10.f);
	}
	else {
		res = _edgeNormals[1] * 2.f * _valence[1];
		res += _edgeNormals[0] * 4.f;
		res += _fringeNormals[2] * 2.f;
		res += _fringeNormals[3] * 2.f;
		res += _fringeNormals[0];
		res += _fringeNormals[1];
		res = res / (2.f * _valence[1] + 10.f);
	}

	res.normalize();
	return res;
}

}