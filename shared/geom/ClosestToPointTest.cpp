/*
 *  ClosestToPointTest.cpp
 *  
 *
 *  Created by jian zhang on 12/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "ClosestToPointTest.h"
#include <math/Plane.h>

namespace aphid {

ClosestToPointTestResult::ClosestToPointTestResult() : 
_hasResult(false) {}

void ClosestToPointTestResult::reset()
{ _hasResult = false; }

void ClosestToPointTestResult::reset(const Vector3F & p, 
									float initialDistance,
									bool fastOp) 
{
	_distance = initialDistance;
	_toPoint = p;
	_hasResult = false;
	_isInside = false;
	_isFast = fastOp;
}

bool ClosestToPointTestResult::closeTo(const BoundingBox & box) const
{ return box.distanceTo(_toPoint) < _distance; }

bool ClosestToPointTestResult::closeEnough() const
{ return _distance < 1e-3f; }

Plane ClosestToPointTestResult::asPlane() const
{ return Plane(_hitNormal, _hitPoint); }

}