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
	void setHasRollCage(bool x);
	void setRollCagePosition(const float & x, const float & y, const float & z);
	void setRollCageDim(const float & x, const float & y, const float & z);
	void setRollCageAngles(const float & aMin, const float & aMax, const float & aSide);
	
	const int numAxis() const;
	const Vector3F origin() const;
	const Vector3F getChassisDim() const;
	const bool hasRollCage() const;
	void getRollCageParam(Vector3F & pos, Vector3F & dim, Vector3F & ang) const;
protected:
	Suspension & suspension(const int & i);
	const Suspension & suspension(const int & i) const;
	Wheel & wheel(const int & i, const int & side);
	const Wheel & wheel(const int & i, const int & side) const;
	const float axisZ(const int & i) const;
	
	const Matrix44F wheelTM(const int & i, bool isLeft = true) const;
	const Vector3F wheelOrigin(const int & i, bool isLeft = true) const;
	void computeDriveZ();
	void computeSteerBase();
	const Vector3F turnAround(const float & ang) const;
	const float wheelSpan(const int & i) const;
private:
	#define MAXNUMAXIS 9
	Vector3F m_origin, m_hullDim;
	Vector3F m_axisCoord[MAXNUMAXIS];
	Suspension m_suspension[MAXNUMAXIS];
	Wheel m_wheel[MAXNUMAXIS][2];
	Vector3F m_rollCagePos, m_rollCageDim, m_rollCageAng;
	float m_driveCenterZ, m_steerBase;
	int m_numAxis;
	bool m_hasRollCage;
};
}