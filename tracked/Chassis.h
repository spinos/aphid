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
	virtual ~Chassis();
	void setOrigin(const Vector3F & p);
	void setSpan(const float & x);
	void setWidth(const float & x);
	void setHeight(const float & x);
	void setTrackWidth(const float & x);
	void setDriveSprocketRadius(const float & x);
	void setTensionerRadius(const float & x);
	void setNumRoadWheels(const int & x);
	void setRoadWheelZ(const int & i, const float & x);
	void setNumSupportRollers(const int & x);
	void setSupportRollerZ(const int & i, const float & x);
	void setTorsionBarLength(const float & x);
	void setTorsionBarSize(const float & x);
	const float torsionBarLength() const;
	const float torsionBarSize() const;
	const float trackWidth() const;
	const float span() const;
	const float driveSprocketRadius() const;
	const float tensionerRadius() const;
	const float roadWheelRadius() const;
	const float supportRollerRadius() const;
	const int numRoadWheels() const;
	const int numSupportRollers() const;
	const Vector3F center() const;
	const Vector3F extends() const;
	const Vector3F trackOrigin(bool isLeft = true) const;
	const Vector3F driveSprocketOrigin(bool isLeft = true) const;
	const Vector3F driveSprocketOriginObject(bool isLeft = true) const;
	const Vector3F tensionerOrigin(bool isLeft = true) const;
	const Vector3F tensionerOriginObject(bool isLeft = true) const;
	const Vector3F roadWheelOrigin(const int & i, bool isLeft = true) const;
	const Vector3F roadWheelOriginObject(const int & i, bool isLeft = true) const;
	const Vector3F supportRollerOrigin(const int & i, bool isLeft = true) const;
	const Vector3F supportRollerOriginObject(const int & i, bool isLeft = true) const;
	const Vector3F torsionBarHingeObject(const int & i, bool isLeft = true) const;
	const Vector3F torsionBarHinge(const int & i, bool isLeft = true) const;
private:
	Vector3F m_origin;
	float m_span, m_width, m_height, m_trackWidth;
	float m_driveSprocketRadius, m_tensionerRadius, m_roadWheelRadius, m_supportRollerRadius;
	float m_tensionerOriginRise, m_roadWheelY, m_supportRollerY, m_torsionBarLength, m_torsionBarSize;
	float * m_roadWheelZ;
	float * m_supportRollerZ;
	int m_numRoadWheels, m_numSupportRollers;
};