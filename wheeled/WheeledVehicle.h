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
	void setGas(const float & x);
	void addGas(const float & x);
	void addSteerAngle(const float & x);
	void setSteerAngle(const float & x);
	void setBrakeStrength(const float & x);
	void addBrakeStrength(const float & x);
	void setGoForward(bool x);
	void setParkingBrake(bool x);
	const Vector3F vehicleVelocity() const;
	const Matrix44F vehicleTM() const;
	const Vector3F vehicleTraverse();
	const float turnAngle() const;
	const float gasStrength() const;
	const float brakeStrength() const;
	const bool goingForward() const;
	void differential(int i, float * dst) const;
	const float drifting() const;
	void wheelForce(int i, float * dst) const;
	void wheelSlip(int i, float * dst) const;
	void wheelSkid(int i, float * dst) const;
	void wheelFriction(int i, float * dst) const;
	const float acceleration() const;
	const float downForce() const;
	void changeGear(int x);
	const int gear() const;
private:
    btCollisionShape* createChassisShape();
    void computeAcceleration();
    void computeDifting();
    void applyDownForce();
private:
    Vector3F m_prevOrigin, m_prevVelocity;
	float m_acceleration, m_drifting, m_downForce;
	float m_gasStrength, m_steerAngle, m_brakeStrength;
	float m_mass;
	int m_gear;
	bool m_parkingBrake;
};
}