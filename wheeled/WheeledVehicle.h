/*
 *  WheeledVehicle.h
 *  wheeled
 *
 *  Created by jian zhang on 5/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <GroupId.h>
#include "WheeledChassis.h"
namespace caterpillar {
class WheeledVehicle : public WheeledChassis, public GroupId {
public:
	WheeledVehicle();
	virtual ~WheeledVehicle();
	void create();
	void update();
	void setTargetSpeed(const float & x);
	void addTargetSpeed(const float & x);
	void addSteerAngle(const float & x);
	void setSteerAngle(const float & x);
	void setBrakeStrength(const float & x);
	void addBrakeStrength(const float & x);
	void displayStatistics();
	const Vector3F vehicleVelocity() const;
	const Matrix44F vehicleTM() const;
	const Vector3F vehicleTraverse();
	const bool goingForward() const;
private:
    Vector3F m_prevOrigin;
	float m_targetSpeed, m_steerAngle, m_brakeStrength;
};
}