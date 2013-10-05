/*
 *  CircleCurve.h
 *  brdf
 *
 *  Created by jian zhang on 9/30/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <BaseCurve.h>

class CircleCurve : public BaseCurve {
public:
	CircleCurve();
	void create();
	virtual ~CircleCurve();
	
	void setRadius(float x);
	
private:
	float m_radius, m_eccentricity;
};