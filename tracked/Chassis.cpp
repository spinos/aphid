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
	m_span = 80.f;
	m_width = 20.f;
	m_height = 10.f;
	m_trackWidth = 8.f;
	m_driveSprocketRadius = 4.f;
	m_tensionerRadius = 4.f;
	m_roadWheelRadius = 4.f;
	m_tensionerOriginRise = 0.f;
	m_roadWheelY = -2.5f;
	m_roadWheelZ = NULL;
	m_numRoadWheels = 0;
	m_supportRollerZ = NULL;
	m_numSupportRollers = 0;
	m_supportRollerY = 3.f;
	m_supportRollerRadius = 1.3f;
}

Chassis::~Chassis()
{
	if(m_roadWheelZ) delete[] m_roadWheelZ;
}

void Chassis::setOrigin(const Vector3F & p) { m_origin = p; }
void Chassis::setSpan(const float & x) { m_span = x; }
void Chassis::setWidth(const float & x) { m_width = x; }
void Chassis::setHeight(const float & x) { m_height = x; }
void Chassis::setTrackWidth(const float & x) { m_trackWidth = x; }
void Chassis::setDriveSprocketRadius(const float & x) { m_driveSprocketRadius = x; }
void Chassis::setTensionerRadius(const float & x) { m_tensionerRadius = x; }

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
	m_supportRollerZ = new float[x];
}

void Chassis::setSupportRollerZ(const int & i, const float & x)
{
	m_supportRollerZ[i] = x;
}

const float Chassis::trackWidth() const { return m_trackWidth; }
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

const Vector3F Chassis::trackOrigin(bool isLeft) const
{
	float d = 1.f;
	if(!isLeft) d = -d;
	Vector3F r = m_origin - Vector3F::ZAxis * m_span * 0.5f + Vector3F::XAxis * ( m_width * .5f * d + m_trackWidth * .51f * d); 
	return r;
}

const Vector3F Chassis::driveSprocketOrigin(bool isLeft) const
{
	return trackOrigin(isLeft);
}

const Vector3F Chassis::driveSprocketOriginObject(bool isLeft) const
{
	return trackOrigin(isLeft) - center();
}

const Vector3F Chassis::tensionerOrigin(bool isLeft) const
{
	return trackOrigin(isLeft) + Vector3F::ZAxis * m_span + Vector3F::YAxis * m_tensionerOriginRise;
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
