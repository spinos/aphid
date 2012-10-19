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
class BoundingBox {
public:
	BoundingBox();
	void reset();
	void setMin(float x, float y, float z);
	void setMax(float x, float y, float z);
	void updateMin(const Vector3F & p);
	void updateMax(const Vector3F & p);
	
	int getLongestAxis() const;
	
	void split(int axis, float pos, BoundingBox & left, BoundingBox & right) const;

	Vector3F m_min, m_max;
};