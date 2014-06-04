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
#include <Common.h>
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
	btRigidBody* chassisBody = PhysicsState::engine->createRigidBody(chassisShape, trans, 128.f);
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
	if(!PhysicsState::engine->isPhysicsEnabled()) return;
	
	float ang = m_steerAngle;
	if(ang < -1.f) ang = -1.f;
	else if(ang > 1.f) ang = 1.f;
	
	const Matrix44F t = vehicleTM();
	Vector3F vel = Vector3F::ZAxis * 8.f;
	vel = t.transformAsNormal(vel);
	vel.normalize();
	vel *= m_targetSpeed;
	
	//const Vector3F vvel = vehicleVelocity();
	//std::cout<<"v "<<vel.x<<" "<<vel.y<<" "<<vel.z<<"\n";
	//std::cout<<"vel "<<vvel.x<<" "<<vvel.y<<" "<<vvel.z<<"\n";
	//std::cout<<"v * vel "<<vel.normal().dot(vvel.normal())<<"\n";
	
	for(int i = 0; i < numAxis(); i++) {
		suspension(i).steer(turnAround(i, ang), wheelSpan(i));
		suspension(i).powerDrive(vel, wheel(i).radius());
	}
}

void WheeledVehicle::addSteerAngle(const float & x) { m_steerAngle += x; }
void WheeledVehicle::setSteerAngle(const float & x) { m_steerAngle = x; }

const Matrix44F WheeledVehicle::vehicleTM() const
{
	if(!PhysicsState::engine->isPhysicsEnabled()) return Matrix44F();
	
	btRigidBody * chassisBody = PhysicsState::engine->getRigidBody(getGroup("chassis")[0]);
	btTransform tm = chassisBody->getWorldTransform();
	return Common::CopyFromBtTransform(tm);
}

const Vector3F WheeledVehicle::vehicleVelocity() const
{
	if(!PhysicsState::engine->isPhysicsEnabled()) return Vector3F::Zero;
	
	btRigidBody * chassisBody = PhysicsState::engine->getRigidBody(getGroup("chassis")[0]);
	const btVector3 chasisVel = chassisBody->getLinearVelocity(); 
	return Vector3F(chasisVel[0], chasisVel[1], chasisVel[2]);
}

void WheeledVehicle::displayStatistics()
{
	if(!PhysicsState::engine->isPhysicsEnabled()) return;
	
	std::cout<<"target velocity: "<<m_targetSpeed<<"\n";
	std::cout<<"turn angle: "<<m_steerAngle<<"\n";
	std::cout<<"vehicle linear velocity: "<<vehicleVelocity().length()<<"\n";
}

}