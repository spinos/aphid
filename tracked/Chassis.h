/*
 *  Chassis.h
 *  tracked
 *
 *  Created by jian zhang on 5/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <AllMath.h>

class Chassis {
public:
	Chassis();
	void setOrigin(const Vector3F & p);
	void setSpan(const float & x);
	void setWidth(const float & x);
	void setHeight(const float & x);
	void setTrackWidth(const float & x);
	void setDriveSprocketRadius(const float & x);
	void setTensionerRadius(const float & x);
	const float trackWidth() const;
	const float span() const;
	const float driveSprocketRadius() const;
	const float tensionerRadius() const;
	const Vector3F center() const;
	const Vector3F extends() const;
	const Vector3F trackOrigin(bool isLeft = true) const;
	const Vector3F driveSprocketOrigin(bool isLeft = true) const;
	const Vector3F driveSprocketOriginObject(bool isLeft = true) const;
	const Vector3F tensionerOrigin(bool isLeft = true) const;
	const Vector3F tensionerOriginObject(bool isLeft = true) const;
private:
	Vector3F m_origin;
	float m_span, m_width, m_height, m_trackWidth;
	float m_driveSprocketRadius, m_tensionerRadius;
	float m_tensionerOriginRise;
};