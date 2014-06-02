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
	m_targetSpeed = 0.f;
	m_steerAngle = 0.f;
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
	btRigidBody* chassisBody = PhysicsState::engine->createRigidBody(chassisShape, trans, 30.f);
	chassisBody->setDamping(0.f, 0.f);
	
	group("chassis").push_back(id);
	
	Suspension::ChassisBody = chassisBody;
	Suspension::ChassisOrigin = origin();
	
	for(int i = 0; i < numAxis(); i++) {
		btRigidBody* hubL = suspension(i).create(wheelOrigin(i));
		btRigidBody* hubR = suspension(i).create(wheelOrigin(i, false), false);
		
		wheel(i).createShape();
		btRigidBody* wheL = wheel(i).create(wheelTM(i));
		btRigidBody* wheR = wheel(i).create(wheelTM(i, false));
		
		suspension(i).connectWheel(hubL, wheL, true);
		suspension(i).connectWheel(hubR, wheR, false);
	}
	
	computeDriveCenterZ();
}

void WheeledVehicle::setTargetSpeed(const float & x) { m_targetSpeed = x; }
void WheeledVehicle::addTargetSpeed(const float & x) { m_targetSpeed += x; }

void WheeledVehicle::update() 
{
	Suspension::SteerAngle = m_steerAngle;
	float ang = m_steerAngle;
	if(ang < -1.f) ang = -1.f;
	else if(ang > 1.f) ang = 1.f;
	
	for(int i = 0; i < numAxis(); i++) {
		suspension(i).steer(turnAround(i, ang), wheelSpan(i));
		suspension(i).powerDrive(m_targetSpeed, wheel(i).radius());
	}
}

void WheeledVehicle::addSteerAngle(const float & x) { m_steerAngle += x; }
void WheeledVehicle::setSteerAngle(const float & x) { m_steerAngle = x; }

void WheeledVehicle::displayStatistics()
{
	if(!PhysicsState::engine->isPhysicsEnabled()) return;
	
	btRigidBody * chassisBody = PhysicsState::engine->getRigidBody(getGroup("chassis")[0]);
	const btVector3 chasisVel = chassisBody->getLinearVelocity(); 
	
	std::cout<<"target velocity: "<<m_targetSpeed<<"\n";
	std::cout<<"turn angle: "<<m_steerAngle<<"\n";
	std::cout<<"vehicle linear velocity: "<<chasisVel.length()<<"\n";
}

}