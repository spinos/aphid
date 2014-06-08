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
	m_gasStrength = 0.f;
	m_steerAngle = 0.f;
	m_brakeStrength = 0.f;
	m_goForward = true;
}

WheeledVehicle::~WheeledVehicle() {}

void WheeledVehicle::create() 
{
	resetGroups();
	m_prevOrigin = origin();
	m_prevVelocity.setZero();
	
	const Vector3F dims = getChassisDim() * .5f; 
	dims.verbose("hulldim");
	btCollisionShape* chassisShape = PhysicsState::engine->createBoxShape(dims.x, dims.y, dims.z);
	
	btTransform trans;
	trans.setIdentity();
	trans.setOrigin(btVector3(origin().x, origin().y, origin().z));
	
	const int id = PhysicsState::engine->numCollisionObjects();
	btRigidBody* chassisBody = PhysicsState::engine->createRigidBody(chassisShape, trans, 200.f);
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

void WheeledVehicle::setGas(const float & x) { m_gasStrength = x; }
void WheeledVehicle::addGas(const float & x) { m_gasStrength += x; if(m_gasStrength > 1.f) m_gasStrength = 1.f; }
void WheeledVehicle::setGoForward(bool x) { m_goForward = x; }
void WheeledVehicle::setBrakeStrength(const float & x) { m_brakeStrength = x; }
void WheeledVehicle::addBrakeStrength(const float & x) { m_brakeStrength += x; if(m_brakeStrength > 1.f) m_brakeStrength = 1.f; }
void WheeledVehicle::addSteerAngle(const float & x) { m_steerAngle += x; }
void WheeledVehicle::setSteerAngle(const float & x) { m_steerAngle = x; }

const float WheeledVehicle::gasStrength() const { return m_gasStrength; }
const float WheeledVehicle::brakeStrength() const { return m_brakeStrength; }
const float WheeledVehicle::turnAngle() const { return m_steerAngle; }
const bool WheeledVehicle::goingForward() const { return m_goForward; }

void WheeledVehicle::update() 
{
	if(!PhysicsState::engine->isPhysicsEnabled()) return;
	
	float ang = m_steerAngle;
	if(ang < -1.f) ang = -1.f;
	else if(ang > 1.f) ang = 1.f;
	
	const Vector3F around = turnAround(ang);
	
	for(int i = 0; i < numAxis(); i++) {
		suspension(i).update();
		
		suspension(i).steer(around, axisZ(i), wheelSpan(i));
		suspension(i).computeDifferential(around, axisZ(i), wheelSpan(i));
		
		suspension(i).drive(m_gasStrength, m_brakeStrength, goingForward());
	}
	
	m_acceleration = vehicleVelocity().length() - m_prevVelocity.length();
	m_prevVelocity = vehicleVelocity();
}

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

const Vector3F WheeledVehicle::vehicleTraverse()
{
    if(!PhysicsState::engine->isPhysicsEnabled()) return Vector3F::Zero;
	const Vector3F cur = vehicleTM().getTranslation();
	const Vector3F r = cur - m_prevOrigin;
	m_prevOrigin = cur;
	return r;
}

void WheeledVehicle::differential(int i, float * dst) const
{
	suspension(i).differential(dst);
}

void WheeledVehicle::wheelForce(int i, float * dst) const
{
	suspension(i).wheelForce(dst);
}

void WheeledVehicle::wheelSlip(int i, float * dst) const
{
	suspension(i).wheelSlip(dst);
}

void WheeledVehicle::wheelSkid(int i, float * dst) const
{
	suspension(i).wheelSkid(dst);
}

const float WheeledVehicle::drifting() const
{
	Vector3F vel = vehicleVelocity(); 
	if(vel.length() < 0.1f) return 0.f;
	vel.normalize();
	Matrix44F space = vehicleTM();
	space.inverse();
	vel = space.transformAsNormal(vel);
	return vel.x;
}

const float WheeledVehicle::acceleration() const
{
	return m_acceleration;
}

}