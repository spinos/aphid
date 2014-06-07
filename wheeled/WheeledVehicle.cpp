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
	m_brakeStrength = 0.f;
}

WheeledVehicle::~WheeledVehicle() {}

void WheeledVehicle::create() 
{
	resetGroups();
	m_prevOrigin = origin();
	
	const Vector3F dims = getChassisDim() * .5f; 
	dims.verbose("hulldim");
	btCollisionShape* chassisShape = PhysicsState::engine->createBoxShape(dims.x, dims.y, dims.z);
	
	btTransform trans;
	trans.setIdentity();
	trans.setOrigin(btVector3(origin().x, origin().y, origin().z));
	
	const int id = PhysicsState::engine->numCollisionObjects();
	btRigidBody* chassisBody = PhysicsState::engine->createRigidBody(chassisShape, trans, 190.f);
	chassisBody->setDamping(0.f, 0.f);
	
	group("chassis").push_back(id);
	
	Suspension::ChassisBody = chassisBody;
	Suspension::ChassisOrigin = origin();
	
	for(int i = 0; i < numAxis(); i++) {
		suspension(i).create(wheelOrigin(i));
		suspension(i).create(wheelOrigin(i, false), false);
		
		wheel(i, 0).create(wheelTM(i), true);
		wheel(i, 1).create(wheelTM(i, false), false);
		
		suspension(i).connectWheel(&wheel(i, 0), true);
		suspension(i).connectWheel(&wheel(i, 1), false);
	}
	
	computeDriveZ();
	computeSteerBase();
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
	vel *= m_targetSpeed * cos(ang);
	
	const bool goForward = m_targetSpeed > 0.f;
	const Vector3F around = turnAround(ang);
	
	const float actvs = vehicleVelocity().length();
	float ts = m_targetSpeed; if(ts < 0.f) ts = -ts;
	
	for(int i = 0; i < numAxis(); i++) {
		suspension(i).update();
		
		suspension(i).steer(around, axisZ(i), wheelSpan(i));
		suspension(i).computeDifferential(around, axisZ(i), wheelSpan(i));
		
		if(ts < actvs) suspension(i).brake(1.f - ts / actvs, goForward);
		else {
			if(m_brakeStrength > 0.f) suspension(i).brake(m_brakeStrength, goForward); 
			else suspension(i).drive(vel, goForward);
		}
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
	
	std::cout<<"\ntarget speed: "<<m_targetSpeed<<"\n";
	std::cout<<"actual speed: "<<vehicleVelocity().length()<<"\n";
	std::cout<<"brake strength: "<<m_brakeStrength<<"\n";
	std::cout<<"turn angle: "<<m_steerAngle<<"\n";
}

const Vector3F WheeledVehicle::vehicleTraverse()
{
    if(!PhysicsState::engine->isPhysicsEnabled()) return Vector3F::Zero;
	const Vector3F cur = vehicleTM().getTranslation();
	const Vector3F r = cur - m_prevOrigin;
	m_prevOrigin = cur;
	return r;
}

void WheeledVehicle::setBrakeStrength(const float & x) { m_brakeStrength = x; }
void WheeledVehicle::addBrakeStrength(const float & x) { m_brakeStrength += x; if(m_brakeStrength > 1.f) m_brakeStrength = 1.f; }

const bool WheeledVehicle::goingForward() const
{
	if(!PhysicsState::engine->isPhysicsEnabled()) return false;
	Matrix44F tm = vehicleTM();
	tm.inverse();
	Vector3F vel = vehicleVelocity();
	vel = tm.transformAsNormal(vel);
	return vel.z > 0.f;
}

}