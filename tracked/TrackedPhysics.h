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
	
protected:
	virtual void clientBuildPhysics();
private:
	void createChassis(Chassis & c);
	void createTread(Tread & t);
	void createDriveSprocket(Chassis & c, btRigidBody * chassisBody, bool isLeft = true);
	void createTensioner(Chassis & c, btRigidBody * chassisBody, bool isLeft = true);
private:
	Tread m_leftTread, m_rightTread;
	Chassis m_chassis;
};

