/*
 *  ClosestToPointTest.h
 *  
 *
 *  Created by jian zhang on 12/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_GEOM_CLOSEST_TO_POINT_TEST_H
#define APH_GEOM_CLOSEST_TO_POINT_TEST_H
#include <math/BarycentricCoordinate.h>
#include <BoundingBox.h>

namespace aphid {

class ClosestToPointTestResult {

public:
	BarycentricCoordinate _bar;
	Vector3F _toPoint;
	Vector3F _hitPoint;
	Vector3F _hitNormal;
	float _contributes[4];
	float _distance;
	unsigned _icomponent;
	unsigned _igeometry;
	bool _hasResult;
	bool _isInside;
	bool _isFast;
	
	ClosestToPointTestResult();
	
	void reset();
	void reset(const Vector3F & p, float initialDistance, bool fastOp = false);
	bool closeTo(const BoundingBox & box) const;
	bool closeEnough() const;
	
};
	
}

#endif