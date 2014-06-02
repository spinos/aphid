/*
 *  WheeledVehicle.cpp
 *  wheeled
 *
 *  Created by jian zhang on 5/30/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "WheeledVehicle.h"
#include <DynamicsSolver.h>
#include <PhysicsState.h>
namespace caterpillar {
WheeledVehicle::WheeledVehicle() 
{
	addGroup("chassis");
	m_targetSpeed = 0;
}

WheeledVehicle::~WheeledVehicle() {}

void WheeledVehicle::create() 
{
	resetGroups();
	
	const Vector3F dims = getChassisDim() * .5f; 
	dims.verbose("hulldim");
	btCollisionShape* chassisShape = PhysicsState::engine->createBoxShape(dims.x, dims.y, dims.z);
	
	btTransform trans;
	trans.setIdentity();
	trans.setOrigin(btVector3(origin().x, origin().y, origin().z));
	
	const int id = PhysicsState::engine->numCollisionObjects();
	btRigidBody* chassisBody = PhysicsState::engine->createRigidBody(chassisShape, trans, 10.f);
	chassisBody->setDamping(0.f, 0.f);
	
	group("chassis").push_back(id);
	
	Suspension::ChassisBody = chassisBody;
	Suspension::ChassisOrigin = origin();
	
	for(int i = 0; i < numAxis(); i++) {std::cout<<"c c";
		btRigidBody* hubL = suspension(i).create(wheelOrigin(i));
		btRigidBody* hubR = suspension(i).create(wheelOrigin(i, false), false);
		
		wheel(i).createShape();std::cout<<"c w";
		btRigidBody* wheL = wheel(i).create(wheelTM(i));
		btRigidBody* wheR = wheel(i).create(wheelTM(i, false));
		
		suspension(i).connectWheel(hubL, wheL, true);
		suspension(i).connectWheel(hubR, wheR, false);
	}
}

void WheeledVehicle::setTargetSpeed(const float & x) { m_targetSpeed = x; }
void WheeledVehicle::addTargetSpeed(const float & x) { m_targetSpeed += x; }

void WheeledVehicle::update() 
{
	for(int i = 0; i < numAxis(); i++)
		suspension(i).powerDrive(m_targetSpeed, wheel(i).radius());
}
}