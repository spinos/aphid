/*
 *  Suspension.cpp
 *  wheeled
 *
 *  Created by jian zhang on 5/31/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Suspension.h"
#include <DynamicsSolver.h>
#include <PhysicsState.h>
#include <Common.h>
namespace caterpillar {
#define SPEEDLIMIT 3.14f
Suspension::Profile::Profile() 
{
	_upperWishboneAngle[0] = -.354f;
	_upperWishboneAngle[1] = .35f;
	_lowerWishboneAngle[0] = -.354f;
	_lowerWishboneAngle[1] = .35f;
	_wheelHubX = .6f;
	_wheelHubR = 1.41f;
	_upperJointY = 2.03f; 
	_lowerJointY = -1.f;
	_steerArmJointZ = 2.f;
	_upperWishboneLength = 3.4f;
	_lowerWishboneLength = 5.7f;
	_upperWishboneTilt = .01f;
	_lowerWishboneTilt = -0.19f;
	_steerable = true;
	_powered = false;
}

float Suspension::RodRadius = .17f;
btRigidBody * Suspension::ChassisBody;
Vector3F Suspension::ChassisOrigin;
Suspension::Suspension() {}
Suspension::~Suspension() {}
void Suspension::setProfile(const Profile & info)
{
	m_profile = info;
}

const float Suspension::width() const
{
	float wishboneU = m_profile._upperWishboneLength * cos(m_profile._upperWishboneTilt);
	const float wishboneL = m_profile._lowerWishboneLength * cos(m_profile._lowerWishboneTilt);
	if(wishboneL > wishboneU) wishboneU = wishboneL;
	return wishboneU + m_profile._wheelHubX;
}

btRigidBody* Suspension::create(const Vector3F & pos, bool isLeft) 
{
	Matrix44F tm;
	if(!isLeft) tm.rotateY(PI);
	
	if(isLeft)
		tm.setTranslation(-m_profile._wheelHubX, 0.f, 0.f);
	else
		tm.setTranslation(m_profile._wheelHubX, 0.f, 0.f);
	
	tm.translate(pos);

	btRigidBody* carrier = createCarrier(tm, isLeft);
	btRigidBody* upperArm = createWishbone(tm, true, isLeft);
	btRigidBody* lowerArm = createWishbone(tm, false, isLeft);
	
	btTransform frmCarrier; frmCarrier.setIdentity();
	frmCarrier.getOrigin()[1] = m_profile._upperJointY;
	
	Matrix33F rot; 
	rot.rotateZ(-m_profile._upperWishboneTilt);
	
	Matrix44F armTM; 
	armTM.setRotation(rot); 
	
	btTransform frmArm = Common::CopyFromMatrix44F(armTM);
	
	const float fra = 0.1f;
	btGeneric6DofConstraint* ball = PhysicsState::engine->constrainBy6Dof(*carrier, *upperArm, frmCarrier, frmArm, true);
	ball->setLinearLowerLimit(btVector3(0.0f, 0.0f,0.0f));
	ball->setLinearUpperLimit(btVector3(0.0f, 0.0f,0.0f));
	ball->setAngularLowerLimit(btVector3(-fra, -1.f, -fra));
	ball->setAngularUpperLimit(btVector3(fra, 1.f, fra));
	/*ball->enableSpring(5, true);
	ball->setStiffness(5, 50.f);
	ball->setDamping(5, 0.5f);
	ball->setEquilibriumPoint(5, 0.f);*/
	
	frmCarrier.getOrigin()[1] = m_profile._lowerJointY;
	
	rot.setIdentity();
	rot.rotateZ(-m_profile._lowerWishboneTilt);
	armTM.setRotation(rot);
	frmArm = Common::CopyFromMatrix44F(armTM);
	
	btGeneric6DofConstraint*ball1 = PhysicsState::engine->constrainBy6Dof(*carrier, *lowerArm, frmCarrier, frmArm, true);
	ball1->setLinearLowerLimit(btVector3(0.0f, 0.0f,0.0f));
	ball1->setLinearUpperLimit(btVector3(0.0f, 0.0f,0.0f));
	ball1->setAngularLowerLimit(btVector3(-fra, -1.f, -fra));
	ball1->setAngularUpperLimit(btVector3(fra, 1.f, fra));
	/*ball1->enableSpring(5, true);
	ball1->setStiffness(5, 50.f);
	ball1->setDamping(5, 0.5f);
	ball1->setEquilibriumPoint(5, 0.f);*/
	
	createSteeringArm(carrier, tm, isLeft);
	
	return carrier;
}

btRigidBody* Suspension::createCarrier(const Matrix44F & tm, bool isLeft)
{
	btCollisionShape* armShape = PhysicsState::engine->createCylinderShape(RodRadius, (m_profile._upperJointY - m_profile._lowerJointY) * .5f, RodRadius);
	btCollisionShape* hubShape = PhysicsState::engine->createCylinderShape(m_profile._wheelHubR, m_profile._wheelHubX * .5f, m_profile._wheelHubR);
	
	btCompoundShape* carrierShape = new btCompoundShape();
	btTransform childT; childT.setIdentity();
	childT.getOrigin()[1] = (m_profile._upperJointY + m_profile._lowerJointY) * .5f;
	carrierShape->addChildShape(childT, armShape);
	
	Matrix44F ctm; 
	if(isLeft) {
		ctm.rotateZ(PI * -.5f);
	}
	else {
		ctm.rotateZ(PI * .5f);
	}
	ctm.translate(Vector3F(m_profile._wheelHubX * .5f, 0.f, 0.f));
	
	childT = Common::CopyFromMatrix44F(ctm);
	carrierShape->addChildShape(childT, hubShape);
	
	btTransform trans = Common::CopyFromMatrix44F(tm);
	btRigidBody* carrierBody = PhysicsState::engine->createRigidBody(carrierShape, trans, 4.f);
	carrierBody->setDamping(0.f, 0.f);
	
	if(isLeft) m_wheelHub[0] = carrierBody;
	else m_wheelHub[1] = carrierBody;
	
	return carrierBody;
}

btRigidBody* Suspension::createWishbone(const Matrix44F & tm, bool isUpper, bool isLeft)
{
	Matrix44F btm;
	
	if(isUpper) btm.rotateZ(m_profile._upperWishboneTilt);
	else btm.rotateZ(m_profile._lowerWishboneTilt);
	if(!isLeft) btm.rotateY(PI);
	
	if(isUpper) btm.translate(Vector3F(0.f, m_profile._upperJointY, 0.f));
	else btm.translate(Vector3F(0.f, m_profile._lowerJointY, 0.f));
	
	btm.translate(tm.getTranslation());
	
	btCompoundShape* wishboneShape = createWishboneShape(isUpper, isLeft);
	
	btTransform trans = Common::CopyFromMatrix44F(btm);
	btRigidBody* wishboneBody = PhysicsState::engine->createRigidBody(wishboneShape, trans, 1.f);
	wishboneBody->setDamping(0.f, 0.f);
	
	connectArm(wishboneBody, btm, isUpper, isLeft, true);
	connectArm(wishboneBody, btm, isUpper, isLeft, false);
	
	return wishboneBody;
}

void Suspension::connectArm(btRigidBody* arm, const Matrix44F & tm, bool isUpper, bool isLeft, bool isFront)
{
	Matrix44F localTM = wishboneHingTMLocal(isUpper, isLeft, isFront);
	
	Matrix33F rot; 
	if(isUpper) rot.rotateZ(-m_profile._upperWishboneTilt);
	else rot.rotateZ(-m_profile._lowerWishboneTilt);
	localTM.setRotation(rot);
	btTransform frmB = Common::CopyFromMatrix44F(localTM);
	
	Matrix44F hingeTM = wishboneHingTMLocal(isUpper, isLeft, isFront);
	hingeTM *= tm;
	
	rot.setIdentity();
	if(!isLeft) rot.rotateY(PI);
	hingeTM.setTranslation(hingeTM.getTranslation() - ChassisOrigin);
	hingeTM.setRotation(rot);	
	btTransform frmA = Common::CopyFromMatrix44F(hingeTM);
	
	btGeneric6DofSpringConstraint* hinge = PhysicsState::engine->constrainBySpring(*ChassisBody, *arm, frmA, frmB, true);
	hinge->setAngularLowerLimit(btVector3(0.0, 0.0, -.2f));
	hinge->setAngularUpperLimit(btVector3(0.0, 0.0, .2f));
	hinge->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
	hinge->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
	
	if(isUpper) return;
	
	hinge->enableSpring(5, true);
	hinge->setStiffness(5, 90.f);
	hinge->setDamping(5, 0.1f);
	hinge->setEquilibriumPoint(5, 0.f);
}

btCompoundShape* Suspension::createWishboneShape(bool isUpper, bool isLeft)
{
	btCompoundShape* shape = new btCompoundShape();
	
	float l, ang;
	wishboneLA(isUpper, isLeft, true, l, ang);
		
	Matrix44F tm;
	tm.rotateZ(PI * .5f);
	tm.rotateY(-ang);
	
	tm.setTranslation(Vector3F(-l * .5f * cos(ang), 0.f, -l * .5f * sin(ang)));
	
	btTransform childT = Common::CopyFromMatrix44F(tm);
	
	btCollisionShape* armShape = PhysicsState::engine->createCylinderShape(RodRadius, l * .5f, RodRadius);
	
	shape->addChildShape(childT, armShape);
	
	wishboneLA(isUpper, isLeft, false, l, ang);
	
	tm.setIdentity();
	tm.rotateZ(PI * .5f);
	tm.rotateY(-ang);
	
	tm.setTranslation(Vector3F(-l * .5f * cos(ang), 0.f, -l * .5f * sin(ang)));
	
	childT = Common::CopyFromMatrix44F(tm);
	
	armShape = PhysicsState::engine->createCylinderShape(RodRadius, l * .5f, RodRadius);
	
	shape->addChildShape(childT, armShape);
	
	return shape;
}

const Matrix44F Suspension::wishboneHingTMLocal(bool isUpper, bool isLeft, bool isFront) const
{	
	float l, ang;
	wishboneLA(isUpper, isLeft, isFront, l, ang);
	
	Matrix44F local;
	local.setTranslation(Vector3F(-l * cos(ang), 0.f, -l  * sin(ang)));
	
	return local;
}

void Suspension::wishboneLA(bool isUpper, bool isLeft, bool isFront, float & l, float & a) const
{
	float lA, lB, angA, angB;
	if(isUpper) {
		angA = m_profile._upperWishboneAngle[0];
		lA = m_profile._upperWishboneLength / cos(angA);
		
		angB = m_profile._upperWishboneAngle[1];
		lB = m_profile._upperWishboneLength / cos(angB);
	}
	else {
		angA = m_profile._lowerWishboneAngle[0];
		lA = m_profile._lowerWishboneLength / cos(angA);
		
		angB = m_profile._lowerWishboneAngle[1];
		lB = m_profile._lowerWishboneLength / cos(angB);
	}
	
	if(!isLeft) {
		angA *= -1.f;
		angB *= -1.f;
	}
	
	l = lA;
	a = angA;
	if(!isFront) {
		l = lB;
		a = angB;
	}
}

btRigidBody* Suspension::createSteeringArm(btRigidBody* carrier, const Matrix44F & tm, bool isLeft)
{
	Matrix44F btm;
	
	btm.rotateZ(m_profile._lowerWishboneTilt);
	if(!isLeft) btm.rotateY(PI);
	
	btm.translate(Vector3F(0.f, 0.f, m_profile._steerArmJointZ));
	
	btm.translate(tm.getTranslation());
	
	float l, ang;
	wishboneLA(false, isLeft, true, l, ang);
	
	Matrix44F atm;
	atm.rotateZ(PI * .5f);

	atm.setTranslation(Vector3F(-l * .5f * cos(ang), 0.f, 0.f));
	
	atm *= btm;
	
	btCollisionShape* armShape = PhysicsState::engine->createCylinderShape(RodRadius, l * .5f * cos(ang), RodRadius);
	
	btTransform trans = Common::CopyFromMatrix44F(atm);
	btRigidBody* armBody = PhysicsState::engine->createRigidBody(armShape, trans, 1.f);
	armBody->setDamping(0.f, 0.f);
	
	Matrix44F tmB; 
	
	if(isLeft) tmB.rotateZ(-m_profile._lowerWishboneTilt - PI* .5f);
	else tmB.rotateZ(- PI* .5f - m_profile._lowerWishboneTilt);
	
	tmB.translate(0.f, l * .5f * cos(ang), 0.f);
	btTransform frmB = Common::CopyFromMatrix44F(tmB);
	
	Matrix44F tmA;
	if(!isLeft) tmA.rotateY(PI);
	
	Vector3F pobj(0.f, l * .5f * cos(ang), 0.f);
	pobj = atm.transform(pobj);
	pobj -= ChassisOrigin;
	tmA.translate(pobj);
	btTransform frmA = Common::CopyFromMatrix44F(tmA);
	
	btGeneric6DofSpringConstraint* hinge = PhysicsState::engine->constrainBySpring(*ChassisBody, *armBody, frmA, frmB, true);
	
	hinge->setAngularLowerLimit(btVector3(-.2f, -.2f, -.2f));
	hinge->setAngularUpperLimit(btVector3(.2f, .2f, .2f));
	hinge->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
	hinge->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
	
	tmA.setIdentity();
	if(isLeft) tmA.translate(0.f, 0.f, m_profile._steerArmJointZ);
	else tmA.translate(0.f, 0.f, -m_profile._steerArmJointZ);
	frmA = Common::CopyFromMatrix44F(tmA);
	
	tmB.translate(0.f, l * -1.f * cos(ang), 0.f);
	frmB = Common::CopyFromMatrix44F(tmB);
	
	hinge = PhysicsState::engine->constrainBySpring(*carrier, *armBody, frmA, frmB, true);
	
	hinge->setAngularLowerLimit(btVector3(-.2f, -.2f, -.2f));
	hinge->setAngularUpperLimit(btVector3(.2f, .2f, .2f));
	hinge->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
	hinge->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
	
	if(isLeft) m_steerJoint[0] = hinge;
	else m_steerJoint[1] = hinge;
	
	return armBody;
}

const float Suspension::wheelHubX() const { return m_profile._wheelHubX; }

void Suspension::connectWheel(btRigidBody* hub, btRigidBody* wheel, bool isLeft)
{
	btTransform frmA; frmA.setIdentity();
	frmA.getOrigin()[0] = wheelHubX();
	
	btTransform frmB; frmB.setIdentity();
	btGeneric6DofConstraint* drv = PhysicsState::engine->constrainBy6Dof(*hub, *wheel, frmA, frmB, true);
	drv->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
	drv->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
	drv->setAngularLowerLimit(btVector3(-SIMD_PI, 0.0, 0.0));
	drv->setAngularUpperLimit(btVector3(SIMD_PI, 0.0, 0.0));
	
	if(isLeft) m_driveJoint[0] = drv;
	else m_driveJoint[1] = drv;
	
	if(isLeft) m_wheel[0] = wheel;
	else m_wheel[1] = wheel;
}

const bool Suspension::isPowered() const { return m_profile._powered; }
const bool Suspension::isSteerable() const { return m_profile._steerable; }

void Suspension::powerDrive(const Vector3F & targetVelocity, const float & wheelR)
{
	const float speed = targetVelocity.length();
	if(speed == 0.f) return applyBrake(true);
	else applyBrake(false);
	
	if(!isPowered()) return;
	limitDrive(0, speed, wheelR);
	limitDrive(1, speed, wheelR);
}

float Suspension::limitDrive(const int & i, const float & targetSpeed, const float & r)
{
	float wheelSpeed = wheelVelocity(i).length();
	float diff = targetSpeed - wheelSpeed;
	
	//float low = (80.f - wheelSpeed) / 80.f;
	//if(low < 0.f) low = 0.f;
	
	const float lmt = r * SPEEDLIMIT;// * (1.f - .5f * low);
	if(diff > lmt) diff = lmt;
	else if(diff < -lmt) diff = -lmt;
	
	wheelSpeed += diff;std::cout<<"limit ["<<i<<"] "<<wheelSpeed;
	const float rps = wheelSpeed / r;
	applyMotor(rps, i);
	return rps;
}

void Suspension::applyBrake(bool enable)
{
	if(enable) {
		applyMotor(0.f, 0);
		applyMotor(0.f, 1);
		return;
	}
	if(!isPowered()) {
		m_driveJoint[0]->getRotationalLimitMotor(0)->m_enableMotor = false;
		m_driveJoint[1]->getRotationalLimitMotor(0)->m_enableMotor = false;
	}
}

void Suspension::applyMotor(float rps, const int & i)
{
	m_driveJoint[i]->getRotationalLimitMotor(0)->m_enableMotor = true;
	if(i==0) m_driveJoint[i]->getRotationalLimitMotor(0)->m_targetVelocity = -rps;
	else m_driveJoint[i]->getRotationalLimitMotor(0)->m_targetVelocity = rps;
	m_driveJoint[i]->getRotationalLimitMotor(0)->m_maxMotorForce = 10.f;
	m_driveJoint[i]->getRotationalLimitMotor(0)->m_damping = 0.5f;
}

void Suspension::steer(const Vector3F & around, const float & wheelSpan)
{
	if(!isSteerable()) return;
	
	const float hspan = wheelSpan * .5f - wheelHubX();
	if(around.x < hspan && around.x > -hspan) {
	    steerWheel(0.f, 0);
	    steerWheel(0.f, 1);
	    return;
	}
	
	float lx = hspan + around.x;
	
	float rx = -hspan + around.x;
	
	steerWheel(atan(around.z / lx), 0);
	steerWheel(atan(around.z / rx), 1);
}

void Suspension::steerWheel(const float & ang, int i)
{
	btTransform & frmA = m_steerJoint[i]->getFrameOffsetA();
	if(i == 0) {
		frmA.getOrigin()[0] = sin(ang) * m_profile._steerArmJointZ;
		frmA.getOrigin()[2] = cos(ang) * m_profile._steerArmJointZ;
	}
	else {
		frmA.getOrigin()[0] = -sin(ang) * m_profile._steerArmJointZ;
		frmA.getOrigin()[2] = -cos(ang) * m_profile._steerArmJointZ;
	}
}

const Matrix44F Suspension::wheelHubTM(const int & i) const
{
	btTransform tm = m_wheelHub[i]->getWorldTransform();
	return Common::CopyFromBtTransform(tm);
}

const Vector3F Suspension::wheelVel(const int & i) const
{
	const btVector3 vel = m_wheel[i]->getLinearVelocity(); 
	return Vector3F(vel[0], vel[1], vel[2]);
}

const Vector3F Suspension::wheelVelocity(const int & i) const
{
	Vector3F vel = wheelVel(i);
	Matrix44F tm = wheelHubTM(i); 
	
	Vector3F hubx(tm.M(0,0), tm.M(0,1),tm.M(0,2));
	btTransform wtm = m_wheel[i]->getWorldTransform();
	Vector3F welx(wtm.getBasis()[0][0], wtm.getBasis()[1][0],wtm.getBasis()[2][0]);
	
	Vector3F front = Vector3F::ZAxis * 100.f;
	if(i > 0) front *= -1.f;
	front = tm.transformAsNormal(front);
	vel *= vel.normal().dot(front.normal());
	return vel;
}

}