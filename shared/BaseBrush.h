/*
 *  BaseBrush.h
 *  mallard
 *
 *  Created by jian zhang on 9/15/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
#include <Ray.h>
class BaseBrush {
public:
	BaseBrush();
	virtual ~BaseBrush();
	void setSpace(const Vector3F & point, const Vector3F & facing);
	void setRadius(float x);
	void setNumDarts(unsigned x);
	
	Matrix44F getSpace() const;
	float getRadius() const;
	unsigned getNumDarts() const;
	Ray getObjectRay(unsigned idx) const;
private:
	char ignoreTooClose(Vector3F p, Vector3F *data, unsigned count, float d) const;
private:
	Matrix44F m_space;
	float m_radius;
	Vector3F * m_darts;
	unsigned m_numDarts;
};