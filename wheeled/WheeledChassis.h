/*
 *  WheeledChassis.h
 *  wheeled
 *
 *  Created by jian zhang on 5/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>
namespace caterpillar {
class WheeledChassis {
public:
	WheeledChassis();
	virtual ~WheeledChassis();

	void setHullDim(const Vector3F & p);
	void setOrigin(const Vector3F & p);
	void setNumAxis(const int & x);
	void setAxisCoord(const int & i, const float & span, const float & y, const float & z);
protected:
	
private:
	#define MAXNUMAXIS 9
	Vector3F m_axisCoord[MAXNUMAXIS];
	Vector3F m_origin, m_hullDim;
	int m_numAxis;
};
}