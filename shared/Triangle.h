/*
 *  Triangle.h
 *  arum
 *
 *  Created by jian zhang on 9/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <math/Vector3F.h>
#include <math/BoundingBox.h>

namespace aphid {

class Triangle {
public:
    Triangle(const Vector3F& a, const Vector3F& b, const Vector3F& c);
	Triangle(const Vector3F& a, const Vector3F& b, const Vector3F& c, const Vector3F& n);
	
	char intersects(const Triangle * another) const;
	char intersects(const Vector3F& origin, const Vector3F& ray, float maxDistance, Vector3F &position, Vector3F &normal) const;
	bool intersect(const BoundingBox & bb) const;
	char frontIntersects(const Vector3F& origin, const Vector3F& ray, float maxDistance, Vector3F &position, Vector3F &normal) const;
	char backIntersects(const Vector3F& origin, const Vector3F& ray, float maxDistance, Vector3F &position, Vector3F &normal) const;
	char closestHit(const Vector3F& origin, Vector3F& dest, float maxDistance) const;
	int classify(const int & axis, const float &pos) const;
	Vector3F getMin() const;
	Vector3F getMax() const;
	float getMin(int axis) const;
	float getMax(int axis) const;
	void expandBBox(BoundingBox & bbox) const;
	Vector3F center() const;
	Vector3F randomOnPlane() const;
	Vector3F normal() const;
	const Vector3F edge(const int & i) const;
	const Vector3F corner(const int & i) const;
	Vector3F p0, p1, p2, nor;
};

}