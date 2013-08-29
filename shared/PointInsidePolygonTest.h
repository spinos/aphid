/*
 *  PointInsidePolygonTest.h
 *  mallard
 *
 *  Created by jian zhang on 8/29/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>

class PointInsidePolygonTest {
public:
	PointInsidePolygonTest();
	bool isPointInside(const Vector3F & px, const Vector3F & nor, Vector3F *vertices, const int nv) const;
	int closestVertex(const Vector3F & px, Vector3F *vertices, const int nv) const;
};
