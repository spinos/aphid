/*
 *  TrackedPhysics.h
 *  tracked
 *
 *  Created by jian zhang on 5/18/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "DynamicsSolver.h"
#include "Tread.h"
#include "Chassis.h"

class TrackedPhysics : public DynamicsSolver
{
public:
	TrackedPhysics();
	virtual ~TrackedPhysics();
	void addTension(const float & x);
protected:
	virtual void clientBuildPhysics();
private:
	class CreateWheelProfile {
	public:
		btRigidBody * connectTo;
		Vector3F worldP, objectP;
		float radius, width, mass;
		bool isLeft;
		btRigidBody * dstBody;
		btGeneric6DofConstraint* dstHinge;
	};
	
	void createChassis(Chassis & c);
	void createTread(Tread & t);
	void createDriveSprocket(Chassis & c, btRigidBody * chassisBody, bool isLeft = true);
	void createTensioner(Chassis & c, btRigidBody * chassisBody, bool isLeft = true);
	void createRoadWheels(Chassis & c, btRigidBody * chassisBody, bool isLeft = true);
	void createSupportRollers(Chassis & c, btRigidBody * chassisBody, bool isLeft = true);
	void createWheel(CreateWheelProfile & profile);
private:
	Tread m_leftTread, m_rightTread;
	Chassis m_chassis;
	btGeneric6DofConstraint* m_tension[2];
	btGeneric6DofConstraint* m_drive[2];
	std::deque<btGeneric6DofConstraint*> m_bearing;
};

