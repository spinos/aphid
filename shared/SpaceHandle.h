/*
 *  SpaceHandle.h
 *  masq
 *
 *  Created by jian zhang on 5/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <AllMath.h>

class SpaceHandle {
public:
	SpaceHandle();
	virtual ~SpaceHandle();
	
	void keepOriginalSpace();
	void spaceMatrix(float m[16]) const;
	Vector3F getCenter() const;
	Vector3F displacement() const;
	void setSize(float val);
	float getSize() const;
	Matrix44F m_space;
	Matrix44F m_space0;
	
private:
	float m_size;
};
