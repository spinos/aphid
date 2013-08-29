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

bool PointInsidePolygonTest::isPointInside(const Vector3F & px, const Vector3F & nor, Vector3F *vertices, const int nv) const
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

int PointInsidePolygonTest::closestVertex(const Vector3F & px, Vector3F *vertices, const int nv) const
{
	int res = 0;
	float minD = 10e8;
	Vector3F vx;
	float mag;
	for(int i = 0; i < nv; i++) {
		vx = vertices[i] - px;
		mag = vx.length();
		
		if(mag < minD) {
			minD = mag;
			res = i;
		}
	}
	return res;
}
//:~