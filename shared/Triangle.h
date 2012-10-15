/*
 *  Triangle.h
 *  arum
 *
 *  Created by jian zhang on 9/13/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <Vector3F.h>
#include <Primitive.h>

class Triangle : public Primitive {
public:
	Triangle(const Vector3F& a, const Vector3F& b, const Vector3F& c, const Vector3F& n);
	virtual ~Triangle();
	char intersects(const Triangle * another) const;
	char intersects(const Vector3F& origin, const Vector3F& ray, float maxDistance, Vector3F &position, Vector3F &normal) const;
	char frontIntersects(const Vector3F& origin, const Vector3F& ray, float maxDistance, Vector3F &position, Vector3F &normal) const;
	char backIntersects(const Vector3F& origin, const Vector3F& ray, float maxDistance, Vector3F &position, Vector3F &normal) const;
	char closestHit(const Vector3F& origin, Vector3F& dest, float maxDistance) const;
	
	virtual void name() const;
	
	Vector3F center() const;
	Vector3F randomOnPlane() const;
	Vector3F normal() const;
	Vector3F p0, p1, p2, nor;
};