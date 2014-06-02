/*
 *  WheeledChassis.cpp
 *  wheeled
 *
 *  Created by jian zhang on 5/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "WheeledChassis.h"
namespace caterpillar {
WheeledChassis::WheeledChassis() 
{
	m_hullDim.set(20.3f, 6.f, 44.3f);
	setAxisCoord(0, 17.f, -2.f, 13.f);
	setAxisCoord(1, 17.f, -2.f, -16.1f);
	m_numAxis = 2;
}

WheeledChassis::~WheeledChassis() {}

void WheeledChassis::setHullDim(const Vector3F & p) { m_hullDim = p; }
void WheeledChassis::setOrigin(const Vector3F & p) { m_origin = p; }
void WheeledChassis::setNumAxis(const int & x) 
{
	m_numAxis = x;
	if(m_numAxis < 2) m_numAxis = 2;
	if(m_numAxis > MAXNUMAXIS) m_numAxis = MAXNUMAXIS;
}

const int WheeledChassis::numAxis() const { return m_numAxis; }

void WheeledChassis::setAxisCoord(const int & i, const float & span, const float & y, const float & z)
{
	m_axisCoord[i].set(span, y, z);
}

void WheeledChassis::setSuspensionInfo(const int & i, const Suspension::Profile & profile)
{
	m_suspension[i].setProfile(profile);
}

void WheeledChassis::setWheelInfo(const int & i, const Wheel::Profile & profile)
{
	m_wheel[i].setProfile(profile);
}

const Vector3F WheeledChassis::getChassisDim() const
{
	Vector3F res = m_hullDim;
	float mxWheelSuspensionL = m_wheel[0].width() * .5f + m_suspension[0].width();
	for(int i = 1; i < numAxis(); i++) {
		const float l = m_wheel[i].width() * .5f + m_suspension[i].width();
		if(mxWheelSuspensionL < l) mxWheelSuspensionL = l;
	}
	
	res.x -= mxWheelSuspensionL * 2.f + .2f;
	return res;
}

const Vector3F WheeledChassis::origin() const { return m_origin; }

Suspension & WheeledChassis::suspension(const int & i) { return m_suspension[i]; }
Wheel & WheeledChassis::wheel(const int & i) { return m_wheel[i]; }

const Matrix44F WheeledChassis::wheelTM(const int & i, bool isLeft) const
{
	Matrix44F tm;
	
	if(!isLeft)
		tm.rotateY(PI);
		
	tm.setTranslation(wheelOrigin(i, isLeft));
	return tm;
}

const Vector3F WheeledChassis::wheelOrigin(const int & i, bool isLeft) const
{
	Vector3F t = m_axisCoord[i] + m_origin;
	if(isLeft) t.x = m_origin.x + m_hullDim.x * .5f- m_wheel[i].width() * .5f;
	else t.x = m_origin.x - m_hullDim.x * .5f + m_wheel[i].width() * .5f;
	return t;
}

void WheeledChassis::computeDriveCenterZ()
{
	m_driveCenterZ = 0.f;
	int numDrv = 0;
	for(int i = 0; i < numAxis(); i++) {
		m_driveCenterZ += m_axisCoord[i].z;
		numDrv++;
	}
	if(numDrv > 0) m_driveCenterZ /= (float)numDrv;
}

const Vector3F WheeledChassis::turnAround(const int & i, const float & ang) const
{
	const float z = m_axisCoord[i].z - m_driveCenterZ;
	if(ang < 10e-4 && ang > -10e-4) return Vector3F(0.f, 0.f, z);
	const float x = z / tan(ang);
	return Vector3F(x, 0.f, z);
}

const float WheeledChassis::wheelSpan(const int & i) const
{
	return m_hullDim.x - m_wheel[i].width();
}

}