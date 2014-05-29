/*
 *  Chassis.cpp
 *  tracked
 *
 *  Created by jian zhang on 5/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Chassis.h"

Chassis::Chassis() 
{
	m_span = 70.75f;
	m_width = 30.55f;
	m_height = 12.5f;
	m_trackWidth = 8.21f;
	m_driveSprocketRadius = 3.8f;
	m_driveSprocketY = -.9f;
	m_driveSprocketZ = -39.8f;
	m_tensionerRadius = 3.8f;
	m_roadWheelRadius = 3.8f;
	m_tensionerY = -.9f;
	m_tensionerZ = 38.4f;
	m_roadWheelY = -6.5f;
	m_roadWheelZ = NULL;
	m_numRoadWheels = 0;
	m_supportRollerZ = NULL;
	m_numSupportRollers = 0;
	m_supportRollerY = 1.5f;
	m_supportRollerRadius = 1.5f;
	m_bogieArmLength = 7.f;
	m_bogieArmWidth = 1.f;
	m_torsionBarRestAngle = .49f;
	m_torsionBarTargetAngle = .67f;
	m_toothWidth = .8;
}

Chassis::~Chassis()
{
	if(m_roadWheelZ) delete[] m_roadWheelZ;
}

void Chassis::setDim(const float & x, const float & y, const float & z)
{
	setWidth(x);
	setHeight(y);
	setSpan(z);
}

void Chassis::setOrigin(const Vector3F & p) { m_origin = p; }
void Chassis::setSpan(const float & x) { m_span = x; }
void Chassis::setWidth(const float & x) { m_width = x; }
void Chassis::setHeight(const float & x) { m_height = x; }
void Chassis::setTrackWidth(const float & x) { m_trackWidth = x; }
void Chassis::setDriveSprocketRadius(const float & x) { m_driveSprocketRadius = x; }
void Chassis::setTensionerRadius(const float & x) { m_tensionerRadius = x; }

void Chassis::setRoadWheelRadius(const float & x) { m_roadWheelRadius = x; }
void Chassis::setSupportRollerRadius(const float & x) { m_supportRollerRadius = x; }

void Chassis::setNumRoadWheels(const int & x)
{
	if(m_roadWheelZ) delete[] m_roadWheelZ;
	m_numRoadWheels = x;
	m_roadWheelZ = new float[x];
}

void Chassis::setRoadWheelZ(const int & i, const float & x)
{
	m_roadWheelZ[i] = x;
}

void Chassis::setNumSupportRollers(const int & x)
{
	if(m_supportRollerZ) delete[] m_supportRollerZ;
	m_numSupportRollers = x;
	if(x < 1) {
	     m_supportRollerZ = NULL;
	     return;
	}
	m_supportRollerZ = new float[x];
}

void Chassis::setSupportRollerZ(const int & i, const float & x)
{
	m_supportRollerZ[i] = x;
}

void Chassis::setDriveSprocketY(const float & x) { m_driveSprocketY = x; }
void Chassis::setTensionerY(const float & x) { m_tensionerY = x; }
void Chassis::setRoadWheelY(const float & x) { m_roadWheelY = x; }
void Chassis::setSupportRollerY(const float & x) { m_supportRollerY = x; }
void Chassis::setToothWidth(const float & x) { m_toothWidth = x; }
const float Chassis::trackWidth() const { return m_trackWidth; }
const float Chassis::tensionerWidth() const { return m_trackWidth - m_toothWidth * 4.f; }
const float Chassis::roadWheelWidth() const { return tensionerWidth(); }
const float Chassis::supportRollerWidth() const { return tensionerWidth() * .6f; }
const float Chassis::span() const { return m_span; }
const float Chassis::driveSprocketRadius() const { return m_driveSprocketRadius; }
const float Chassis::tensionerRadius() const { return m_tensionerRadius; }
const float Chassis::roadWheelRadius() const { return m_roadWheelRadius; }
const float Chassis::supportRollerRadius() const { return m_supportRollerRadius; }
const int Chassis::numRoadWheels() const { return m_numRoadWheels; }
const int Chassis::numSupportRollers() const { return m_numSupportRollers; }

const Vector3F Chassis::center() const
{
	return m_origin;
}

const Vector3F Chassis::extends() const 
{
	return Vector3F(m_width, m_height, m_span);
}

const Vector3F Chassis::driveSprocketOrigin(bool isLeft) const
{
	float d = 1.f;
	if(!isLeft) d = -d;
	Vector3F r = m_origin + Vector3F::ZAxis * m_driveSprocketZ + Vector3F::XAxis * ( m_width * .5f * d + m_trackWidth * .51f * d) + Vector3F::YAxis * m_driveSprocketY; 
	return r;
}

const Vector3F Chassis::driveSprocketOriginObject(bool isLeft) const
{
	return driveSprocketOrigin(isLeft) - center();
}

const Vector3F Chassis::tensionerOrigin(bool isLeft) const
{
	float d = 1.f;
	if(!isLeft) d = -d;
	Vector3F r = m_origin + Vector3F::ZAxis * m_tensionerZ + Vector3F::XAxis * ( m_width * .5f * d + m_trackWidth * .51f * d) + Vector3F::YAxis * m_tensionerY; 
	return r;
}

const Vector3F Chassis::tensionerOriginObject(bool isLeft) const
{
	return tensionerOrigin(isLeft) - center();
}

const Vector3F Chassis::roadWheelOrigin(const int & i, bool isLeft) const
{
	float d = 1.f;
	if(!isLeft) d = -d;
	return m_origin + Vector3F::YAxis * m_roadWheelY + Vector3F::ZAxis * m_roadWheelZ[i] + Vector3F::XAxis * ( m_width * .5f * d + m_trackWidth * .51f * d);
}

const Vector3F Chassis::roadWheelOriginObject(const int & i, bool isLeft) const
{
	return roadWheelOrigin(i, isLeft) - center();
}

const Vector3F Chassis::supportRollerOrigin(const int & i, bool isLeft) const
{
	return m_origin + supportRollerOriginObject(i, isLeft);
}

const Vector3F Chassis::supportRollerOriginObject(const int & i, bool isLeft) const
{
	float d = 1.f;
	if(!isLeft) d = -d;
	return Vector3F::YAxis * m_supportRollerY + Vector3F::ZAxis * m_supportRollerZ[i] + Vector3F::XAxis * ( m_width * .5f * d + m_trackWidth * .51f * d);
}

void Chassis::setBogieArmLength(const float & x) { m_bogieArmLength = x; }
void Chassis::setBogieArmWidth(const float & x) { m_bogieArmWidth = x; }
const float Chassis::bogieArmLength() const { return m_bogieArmLength; }
const float Chassis::bogieArmWidth() const { return m_bogieArmWidth; }

const Vector3F Chassis::torsionBarHingeObject(const int & i, bool isLeft) const
{
	return torsionBarHinge(i, isLeft) - m_origin;
}

const Vector3F Chassis::torsionBarHinge(const int & i, bool isLeft) const
{
    const Matrix44F mat = bogieArmOrigin(i, isLeft);
    Vector3F p = mat.getTranslation();
    p.z += 0.5f * m_bogieArmLength * cos(m_torsionBarRestAngle);
    p.y += 0.5f * m_bogieArmLength * sin(m_torsionBarRestAngle);
	return p;
}

void Chassis::setTorsionBarRestAngle(const float & x) { m_torsionBarRestAngle = x; }
const float Chassis::torsionBarRestAngle() const { return m_torsionBarRestAngle; }
void Chassis::setTorsionBarTargetAngle(const float & x) { m_torsionBarTargetAngle = x; }
const float Chassis::torsionBarTargetAngle() const { return m_torsionBarTargetAngle; }

const Vector3F Chassis::computeWheelOrigin(const float & chassisWidth, const float & trackWidth, const float & y, const float & z, bool isLeft) const
{
	float d = 1.f;
	if(!isLeft) d = -d;
	
	Vector3F p(0.f, y, z);
	p.x = d * (chassisWidth + trackWidth) * .5f;
	return p;
}

const bool Chassis::isBackdrive() const { return m_driveSprocketZ < m_tensionerZ; }

const Matrix44F  Chassis::bogieArmOrigin(const int & i, bool isLeft) const
{
    Matrix44F res;
    res.rotateX(m_torsionBarRestAngle);
    Vector3F cen = roadWheelOrigin(i, isLeft);
    float d = 1.f;
	if(!isLeft) d = -d;
	cen.x += -m_trackWidth * .5f * d + m_bogieArmWidth * .7f * d;
    cen.z += 0.5f * m_bogieArmLength * cos(m_torsionBarRestAngle);
    cen.y += 0.5f * m_bogieArmLength * sin(m_torsionBarRestAngle);
    res.setTranslation(cen);
    return res;
}

const Vector3F Chassis::roadWheelOriginToBogie(bool isLeft) const
{
    float d = 1.f;
	if(!isLeft) d = -d;
    return Vector3F::ZAxis * -0.5f * m_bogieArmLength + Vector3F::XAxis * (m_trackWidth * .5f * d - m_bogieArmWidth * .5f * d);
}

const float Chassis::toothWidth() const { return m_toothWidth; }

const Matrix44F Chassis::computeBogieArmOrigin(const float & chassisWidth, const Vector3F & wheelP, const float & l, const float & s, const float & ang) const
{
    Matrix44F tm;
    tm.rotateX(-ang);
    Vector3F p;
    p.x = chassisWidth * .5f + s * .5f;
    p.y = wheelP.y + l * .5f * sin(ang);
    p.z = wheelP.z + l * .5f * cos(ang);
    tm.setTranslation(p);
    return tm;
}

void Chassis::setDriveSprocketZ(const float & x) { m_driveSprocketZ = x; }
void Chassis::setTensionerZ(const float & x) { m_tensionerZ = x; }

const bool Chassis::aroundFirstSupportRoller(Vector3F & p, float & r, bool isLeft) const
{
    if(numSupportRollers() < 1) return false;
	
	Vector3F fp; float fr;
	getFrontWheel(fp, fr);
	if(fp.y + fr > supportRollerOrigin(0, isLeft).y + supportRollerRadius())
		return false;
		
	p = supportRollerOrigin(0, isLeft);
	r = supportRollerRadius();
	return true;
}

const bool Chassis::aroundLastSupportRoller(Vector3F & p, float & r, bool isLeft) const
{
    if(numSupportRollers() < 2) return false;
	
	Vector3F bp; float br;
	getBackWheel(bp, br);
	if(bp.y + br > supportRollerOrigin(numSupportRollers() - 1, isLeft).y + supportRollerRadius())
		return false;
		
	p = supportRollerOrigin(numSupportRollers() - 1, isLeft);
	r = supportRollerRadius();
	return true;
}

void Chassis::getBackWheel(Vector3F & p, float & r, bool isLeft) const
{
	if(isBackdrive()) {
		p = driveSprocketOrigin(isLeft);
		r = driveSprocketRadius();
		return;
	}
	p = tensionerOrigin(isLeft);
	r = tensionerRadius();
}

void Chassis::getFrontWheel(Vector3F & p, float & r, bool isLeft) const
{
	if(isBackdrive()) {
		p = tensionerOrigin(isLeft);
		r = tensionerRadius();
		return;
	}
	p = driveSprocketOrigin(isLeft);
	r = driveSprocketRadius();
}
