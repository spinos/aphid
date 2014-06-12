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
	addGroup("wheel0");
	addGroup("wheel1");
	addGroup("wheel2");
	addGroup("wheel3");
	addGroup("wheel4");
	addGroup("wheel5");
	addGroup("wheel6");
	addGroup("wheel7");
	addGroup("wheel8");
	m_gasStrength = 0.f;
	m_steerAngle = 0.f;
	m_brakeStrength = 0.f;
	m_gear = 0;
	m_parkingBrake = false;
	m_downForce = 0.f;
	m_mass = 200.f;
	m_downForceCoef = 1.f;
}

WheeledVehicle::~WheeledVehicle() {}

void WheeledVehicle::create() 
{
	resetGroups();
	m_prevOrigin = origin();
	m_prevVelocity.setZero();
	
	btCollisionShape* chassisShape = createChassisShape();
	
	const Vector3F ori = origin();
	btTransform trans;
	trans.setIdentity();
	trans.setOrigin(btVector3(ori.x, ori.y, ori.z));
	
	const int id = PhysicsState::engine->numCollisionObjects();
	btRigidBody* chassisBody = PhysicsState::engine->createRigidBody(chassisShape, trans, m_mass);
	chassisBody->setDamping(0.f, 0.f);
	
	group("chassis").push_back(id);
	
	Suspension::ChassisBody = chassisBody;
	Suspension::ChassisOrigin = origin();
	
	for(int i = 0; i < numAxis(); i++) {
		suspension(i).create(wheelOrigin(i));
		suspension(i).create(wheelOrigin(i, false), false);
		
		const std::string wgn = wheelGrpName(i);
		
		wheel(i, 0).create(wheelTM(i), true);
		group(wgn).push_back(wheel(i, 0).rigidBodyId());
		
		wheel(i, 1).create(wheelTM(i, false), false);
		group(wgn).push_back(wheel(i, 1).rigidBodyId());
		
		suspension(i).connectWheel(&wheel(i, 0), true);
		suspension(i).connectWheel(&wheel(i, 1), false);
	}
	
	computeDriveZ();
	computeSteerBase();
}

btCollisionShape* WheeledVehicle::createChassisShape()
{
    const Vector3F dimH = getChassisDim() * .5f; 
    btBoxShape* box = PhysicsState::engine->createBoxShape(dimH.x, dimH.y, dimH.z);
    if(!hasRollCage()) return box;
    
    btCompoundShape* comp = new btCompoundShape();
    btTransform frm; frm.setIdentity();
    comp->addChildShape(frm, box);
    
    Vector3F pos, dim, ang;
    getRollCageParam(pos, dim, ang);
    
    const float barR = .31f;
    const float xfac = dim.y * tan(ang.z);
    Matrix44F tm;
    
    btCylinderShape * bar = PhysicsState::engine->createCylinderShape(barR, dim.x * .5f - xfac, barR);
    
    tm.setIdentity();
    tm.rotateZ(PI * .5f);
    tm.setTranslation(pos.x, pos.y + dim.y * .5f, pos.z - dim.z * .5f);
    frm = Common::CopyFromMatrix44F(tm);
    comp->addChildShape(frm, bar);
    
    tm.setTranslation(pos.x, pos.y + dim.y * .5f, pos.z + dim.z * .5f);
    frm = Common::CopyFromMatrix44F(tm);
    comp->addChildShape(frm, bar);
    
    const float zfac0 = dim.y * tan(ang.x);
    const float l0 = sqrt(dim.y * dim.y + xfac * xfac + zfac0 * zfac0);
    
    bar = PhysicsState::engine->createCylinderShape(barR, l0 * .5f, barR);
    
    Matrix44F rotx; rotx.rotateX(-ang.x);
    
    tm.setIdentity();
    tm *= rotx;
    tm.rotateZ(ang.z);
    tm.setTranslation(pos.x + dim.x * .5f - xfac * .5f, pos.y, pos.z + dim.z * .5f + zfac0 * .5f);
    
    frm = Common::CopyFromMatrix44F(tm);
    comp->addChildShape(frm, bar);
    
    tm.setIdentity();
    tm *= rotx;
    tm.rotateZ(-ang.z);
    tm.setTranslation(pos.x - dim.x * .5f + xfac * .5f, pos.y, pos.z + dim.z * .5f + zfac0 * .5f);
    
    frm = Common::CopyFromMatrix44F(tm);
    comp->addChildShape(frm, bar);
    
    const float zfac1 = dim.y * tan(ang.y);
    const float l1 = sqrt(dim.y * dim.y + xfac * xfac + zfac1 * zfac1);
    
    bar = PhysicsState::engine->createCylinderShape(barR, l1 * .5f, barR);
    
    rotx.setIdentity(); rotx.rotateX(ang.y);
    
    tm.setIdentity();
    tm *= rotx;
    tm.rotateZ(ang.z);
    tm.setTranslation(pos.x + dim.x * .5f - xfac * .5f, pos.y, pos.z - dim.z * .5f - zfac1 * .5f);
    
    frm = Common::CopyFromMatrix44F(tm);
    comp->addChildShape(frm, bar);
    
    tm.setIdentity();
    tm *= rotx;
    tm.rotateZ(-ang.z);
    tm.setTranslation(pos.x - dim.x * .5f + xfac * .5f, pos.y, pos.z - dim.z * .5f - zfac1 * .5f);
    
    frm = Common::CopyFromMatrix44F(tm);
    comp->addChildShape(frm, bar);
    
    PhysicsState::engine->addCollisionShape(comp);
	return comp;
}

void WheeledVehicle::setGas(const float & x) { m_gasStrength = x; }
void WheeledVehicle::addGas(const float & x) { m_gasStrength += x; if(m_gasStrength > 1.f) m_gasStrength = 1.f; }
void WheeledVehicle::setBrakeStrength(const float & x) { m_brakeStrength = x; }
void WheeledVehicle::addBrakeStrength(const float & x) { m_brakeStrength += x; if(m_brakeStrength > 1.f) m_brakeStrength = 1.f; }
void WheeledVehicle::addSteerAngle(const float & x) { m_steerAngle += x; }
void WheeledVehicle::setSteerAngle(const float & x) { m_steerAngle = x; }
void WheeledVehicle::setParkingBrake(bool x) { m_parkingBrake = x; }

const float WheeledVehicle::gasStrength() const { return m_gasStrength; }
const float WheeledVehicle::brakeStrength() const { return m_brakeStrength; }
const float WheeledVehicle::turnAngle() const { return m_steerAngle; }
const bool WheeledVehicle::goingForward() const { return m_gear != 6; }

void WheeledVehicle::update() 
{
	if(!PhysicsState::engine->isPhysicsEnabled()) return;
	
	Suspension::Gear = gear();
	
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
	
	if(m_parkingBrake) suspension(numAxis() - 1).parkingBrake();
	
	computeAcceleration();
	computeDifting();
	applyDownForce();
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

void WheeledVehicle::wheelFriction(int i, float * dst) const
{
    suspension(i).wheelFriction(dst);
}

void WheeledVehicle::computeAcceleration()
{
    const Vector3F curV = vehicleVelocity();
    m_acceleration = curV.length() - m_prevVelocity.length();
	m_prevVelocity = curV;
}

void WheeledVehicle::computeDifting()
{
 	Vector3F vel = vehicleVelocity(); 
	if(vel.length() < 0.1f) return;
	vel.normalize();
	Matrix44F space = vehicleTM();
	space.inverse();
	vel = space.transformAsNormal(vel);
	m_drifting = vel.x;   
}

const float WheeledVehicle::drifting() const
{
	return m_drifting;
}

const float WheeledVehicle::acceleration() const
{
	return m_acceleration;
}

void WheeledVehicle::changeGear(int x) 
{ 
	m_gear += x; 
	if(m_gear < 0) m_gear = 6;
	m_gear = m_gear % 7; 
}

const int WheeledVehicle::gear() const { return m_gear; }

const float WheeledVehicle::downForce() const { return m_downForce; }

void WheeledVehicle::applyDownForce()
{
    if(!PhysicsState::engine->isPhysicsEnabled()) return;
    
    Vector3F vel = m_prevVelocity; 
	if(vel.length() < 0.1f) {
	    m_downForce = 0.f;
	    return;
	}

	Matrix44F space = vehicleTM();
	
	Vector3F down = space.transformAsNormal(Vector3F(0.f, -1.f, 0.f));
	space.inverse();
	vel = space.transformAsNormal(vel);
	float facing = vel.z;
	if(facing < 0.f) facing = -facing;
	
	m_downForce = facing * getChassisDim().x * getChassisDim().y * .0002f * m_downForceCoef;
	if(m_downForce > 1.f) m_downForce = 1.f;
	down *= m_downForce * m_mass * 9.8f;
	btRigidBody * chassisBody = PhysicsState::engine->getRigidBody(getGroup("chassis")[0]);
	chassisBody->applyCentralForce(btVector3(down.x, down.y, down.z));
}

const Vector3F WheeledVehicle::velocity() const { return m_prevVelocity; }
void WheeledVehicle::setDownForceCoef(const float & x) { m_downForceCoef = x; }

const std::string WheeledVehicle::wheelGrpName(int i) const
{
    std::stringstream sst;
    sst.str("");
    sst<<"wheel"<<i;
    return sst.str();
}

}