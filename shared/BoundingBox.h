/*
 *  BoundingBox.h
 *  kdtree
 *
 *  Created by jian zhang on 10/17/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <Vector3F.h>
class Ray;
class BoundingBox {
public:
	BoundingBox();
	void reset();
	void setMin(float x, float y, float z);
	void setMax(float x, float y, float z);
	void updateMin(const Vector3F & p);
	void updateMax(const Vector3F & p);
	
	int getLongestAxis() const;
	const float getMin(int axis) const;
	const float getMax(int axis) const;
	const float area() const;
	const float crossSectionArea(const int &axis) const;
	const float distance(const int &axis) const;
	
	void split(int axis, float pos, BoundingBox & left, BoundingBox & right) const;
	void expandBy(const BoundingBox &another);
	
	void expand(float v);
	
	char intersect(const Ray &ray, float *hitt0, float *hitt1) const;
	char isPointInside(const Vector3F & p) const;
	
	float min(int axis) const;
	float max(int axis) const;
	float m_data[6];
};