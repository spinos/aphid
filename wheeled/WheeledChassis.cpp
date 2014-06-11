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
	m_hullDim.set(20.3f, 4.f, 44.3f);
	setAxisCoord(0, 17.f, -1.f, 13.f);
	setAxisCoord(1, 17.f, -1.f, -16.1f);
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
	m_wheel[i][0].setProfile(profile);
	m_wheel[i][1].setProfile(profile);
}

const Vector3F WheeledChassis::getChassisDim() const
{
	Vector3F res = m_hullDim;
	float mxWheelSuspensionL = wheel(0, 0).width() * .5f + m_suspension[0].width();
	for(int i = 1; i < numAxis(); i++) {
		const float l = wheel(i, 0).width() * .5f + m_suspension[i].width();
		if(mxWheelSuspensionL < l) mxWheelSuspensionL = l;
	}
	
	res.x -= mxWheelSuspensionL * 2.f + .2f;
	return res;
}

const Vector3F WheeledChassis::origin() const { return m_origin; }

Suspension & WheeledChassis::suspension(const int & i) { return m_suspension[i]; }
const Suspension & WheeledChassis::suspension(const int & i) const {return m_suspension[i];}
Wheel & WheeledChassis::wheel(const int & i, const int & side) { return m_wheel[i][side]; }
const Wheel & WheeledChassis::wheel(const int & i, const int & side) const { return m_wheel[i][side]; }

const Matrix44F WheeledChassis::wheelTM(const int & i, bool isLeft) const
{
	Matrix44F tm;
	
	if(!isLeft)
		tm.rotateY(PI);
		
	tm.setTranslation(wheelOrigin(i, isLeft));std::cout<<"arm "<<isLeft<<" "<<tm.getTranslation().str()<<"\n";
	return tm;
}

const Vector3F WheeledChassis::wheelOrigin(const int & i, bool isLeft) const
{
	Vector3F t = m_axisCoord[i] + m_origin;
	if(isLeft) t.x = m_origin.x + m_hullDim.x * .5f- wheel(i, 0).width() * .5f;
	else t.x = m_origin.x - m_hullDim.x * .5f + wheel(i, 0).width() * .5f;
	return t;
}

void WheeledChassis::computeDriveZ()
{
	m_driveCenterZ = 0.f;
	int numDrv = 0;
	for(int i = 0; i < numAxis(); i++) {
		if(!suspension(i).isSteerable()) {
		    m_driveCenterZ += m_axisCoord[i].z;
		    numDrv++;
		}
	}
	if(numDrv > 0) {
		m_driveCenterZ /= (float)numDrv;
		return;
	}
	
	m_driveCenterZ = 0.f;
	for(int i = 0; i < numAxis(); i++) {
		m_driveCenterZ += m_axisCoord[i].z;
	}
	
	m_driveCenterZ /= (float)numAxis();
}

void WheeledChassis::computeSteerBase()
{
	float steerZ = 0.f;
	if(suspension(0).isSteerable())
		steerZ = m_axisCoord[0].z;
		
	if(suspension(numAxis() - 1).isSteerable()) {
		float a = m_axisCoord[numAxis() - 1].z;
		if(a < 0) a = -a;
		if(a > steerZ)
			steerZ = m_axisCoord[numAxis() - 1].z;
	}
		
	m_steerBase = steerZ - m_driveCenterZ;
}

const Vector3F WheeledChassis::turnAround(const float & ang) const
{
	if(ang < .001f && ang > -.001f) return Vector3F(0.f, 0.f, m_driveCenterZ);
	
	return Vector3F(m_steerBase / tan(ang), 0.f, m_driveCenterZ);
}

const float WheeledChassis::wheelSpan(const int & i) const
{
	return m_hullDim.x - wheel(i, 0).width();
}

const float WheeledChassis::axisZ(const int & i) const { return m_axisCoord[i].z;}
}