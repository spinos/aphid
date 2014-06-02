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
#include "Suspension.h"
#include "Wheel.h"
namespace caterpillar {
class WheeledChassis {
public:
	WheeledChassis();
	virtual ~WheeledChassis();

	void setHullDim(const Vector3F & p);
	void setOrigin(const Vector3F & p);
	void setNumAxis(const int & x);
	void setAxisCoord(const int & i, const float & span, const float & y, const float & z);
	void setSuspensionInfo(const int & i, const Suspension::Profile & profile);
	void setWheelInfo(const int & i, const Wheel::Profile & profile);
	
	const int numAxis() const;
	const Vector3F origin() const;
	const Vector3F getChassisDim() const; 
protected:
	Suspension & suspension(const int & i);
	Wheel & wheel(const int & i);
	
	const Matrix44F wheelTM(const int & i, bool isLeft = true) const;
	const Vector3F wheelOrigin(const int & i, bool isLeft = true) const;
	void computeDriveCenterZ();
	const Vector3F turnAround(const int & i, const float & ang) const;
	const float wheelSpan(const int & i) const;
private:
	#define MAXNUMAXIS 9
	Vector3F m_origin, m_hullDim;
	Vector3F m_axisCoord[MAXNUMAXIS];
	Suspension m_suspension[MAXNUMAXIS];
	Wheel m_wheel[MAXNUMAXIS];
	float m_driveCenterZ;
	int m_numAxis;
};
}