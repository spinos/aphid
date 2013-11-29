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
#include <Patch.h>
#include <Ray.h>
class PointInsidePolygonTest : public Patch {
public:
	PointInsidePolygonTest();
	PointInsidePolygonTest(const Vector3F & p0, const Vector3F & p1, const Vector3F & p2, const Vector3F & p3);
	bool isPointInside(const Vector3F & px) const;
	float distanceTo(const Vector3F & origin, Vector3F & closestP, char & inside) const;
	static bool isPointInside(const Vector3F & px, const Vector3F & nor, Vector3F *vertices, const int nv);
	bool intersect(Ray & ray, Vector3F & closestP);
};
