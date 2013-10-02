/*
 *  BoundingRectangle.h
 *  mallard
 *
 *  Created by jian zhang on 10/3/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <Vector2F.h>
class BoundingRectangle {
public:
	BoundingRectangle();
	
	void reset();
	void update(const Vector2F & p);
	void updateMin(const Vector2F & p);
	void updateMax(const Vector2F & p);
	void translate(const Vector2F & d);
	
	const float getMin(int axis) const;
	const float getMax(int axis) const;
	const float distance(const int &axis) const;
	
	bool isPointInside(const Vector2F & p) const;
	
private:
	float m_data[4];
};

