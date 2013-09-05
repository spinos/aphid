/*
 *  PointInsidePolygonTest.cpp
 *  mallard
 *
 *  Created by jian zhang on 8/29/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "PointInsidePolygonTest.h"

PointInsidePolygonTest::PointInsidePolygonTest() {}

PointInsidePolygonTest::PointInsidePolygonTest(const Vector3F & p0, const Vector3F & p1, const Vector3F & p2, const Vector3F & p3) : Patch (p0, p1, p2, p3) {}

bool PointInsidePolygonTest::isPointInside(const Vector3F & px) const
{
	Vector3F nor;
	getNormal(nor);
	float theta = 0.f;
	int j;
	Vector3F v1, v2, axis;
	float mag1, mag2, magx;
	for(int i = 0; i < 4; i++) {
		v1 = vertex(i) - px;
		mag1 = v1.length();
		if(mag1 < EPSILON) return true;
		v1 /= mag1;
		
		j = (i + 1) % 4;
		v2 = vertex(j) - px;
		
		mag2 = v2.length();
		if(mag2 < EPSILON) return true;
		v2 /= mag2;
		
		axis = v1.cross(v2);
		magx = axis.length();
		
		if(magx < EPSILON) return true;
		axis /= magx;
		
		theta += acos(v1.dot(v2)) * axis.dot(nor);
	}
	
	return theta > PI;
}

float PointInsidePolygonTest::distanceTo(const Vector3F & origin, Vector3F & closestP) const
{
	Vector3F px;
	projectPoint(origin, px);
	
	if(isPointInside(px)) {
		closestP = px;
		return Vector3F(px, origin).length();
	}
	
	planarDistanceTo(px, closestP);
	
	return Vector3F(closestP, origin).length();
}

bool PointInsidePolygonTest::isPointInside(const Vector3F & px, const Vector3F & nor, Vector3F *vertices, const int nv)
{
	float theta = 0.f;
	int j;
	Vector3F v1, v2, axis;
	float mag1, mag2, magx;
	for(int i = 0; i < nv; i++) {
		v1 = vertices[i] - px;
		mag1 = v1.length();
		if(mag1 < EPSILON) return true;
		v1 /= mag1;
		
		j = (i + 1) % nv;
		v2 = vertices[j] - px;
		
		mag2 = v2.length();
		if(mag2 < EPSILON) return true;
		v2 /= mag2;
		
		axis = v1.cross(v2);
		magx = axis.length();
		
		if(magx < EPSILON) return true;
		axis /= magx;
		
		theta += acos(v1.dot(v2)) * axis.dot(nor);
	}
	
	return theta > PI;
}

//:~