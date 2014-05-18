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
	m_trackWidth = 4.f;
	m_driveSprocketRadius = 4.f;
	m_tensionerRadius = 4.f;
	m_tensionerOriginRise = 1.f;
}

void Chassis::setOrigin(const Vector3F & p) { m_origin = p; }
void Chassis::setSpan(const float & x) { m_span = x; }
void Chassis::setWidth(const float & x) { m_width = x; }
void Chassis::setHeight(const float & x) { m_height = x; }
void Chassis::setTrackWidth(const float & x) { m_trackWidth = x; }
void Chassis::setDriveSprocketRadius(const float & x) { m_driveSprocketRadius = x; }
void Chassis::setTensionerRadius(const float & x) { m_tensionerRadius = x; }
const float Chassis::trackWidth() const { return m_trackWidth; }
const float Chassis::span() const { return m_span; }
const float Chassis::driveSprocketRadius() const { return m_driveSprocketRadius; }
const float Chassis::tensionerRadius() const { return m_tensionerRadius; }
const Vector3F Chassis::center() const
{
	return Vector3F(m_origin.x, m_origin.y, m_origin.z + m_span * 0.5f);
}

const Vector3F Chassis::extends() const 
{
	return Vector3F(m_width, m_height, m_span);
}

const Vector3F Chassis::trackOrigin(bool isLeft) const
{
	float d = 1.f;
	if(!isLeft) d = -d;
	Vector3F r = m_origin + Vector3F::XAxis * ( m_width * .5f * d + m_trackWidth * .51f * d); 
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
