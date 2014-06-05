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
#define SPEEDLIMIT 2.14f
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
	_damperY = 4.f;
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
	createWishbone(carrier, tm, true, isLeft);
	btRigidBody* lowerArm = createWishbone(carrier, tm, false, isLeft);
	createDamper(lowerArm, tm);
	
	createSteeringArm(carrier, tm, isLeft);
	
	btRigidBody * bar = createSwayBar(tm, lowerArm, isLeft);
	
	if(isLeft) m_swayBarLeft = bar;
	else connectSwayBar(tm, bar);
	
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

btRigidBody* Suspension::createWishbone(btRigidBody* carrier, const Matrix44F & tm, bool isUpper, bool isLeft)
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
	
	btTransform frmCarrier; frmCarrier.setIdentity();
	if(isUpper) frmCarrier.getOrigin()[1] = m_profile._upperJointY;
	else frmCarrier.getOrigin()[1] = m_profile._lowerJointY;
	
	Matrix33F rot; 
	if(isUpper) rot.rotateZ(-m_profile._upperWishboneTilt);
	else rot.rotateZ(-m_profile._lowerWishboneTilt);
	
	Matrix44F armTM; 
	armTM.setRotation(rot); 
	
	btTransform frmArm = Common::CopyFromMatrix44F(armTM);
	
	const float fra = 0.2f;
	btGeneric6DofConstraint* ball = PhysicsState::engine->constrainBy6Dof(*carrier, *wishboneBody, frmCarrier, frmArm, true);
	ball->setLinearLowerLimit(btVector3(0.0f, 0.0f,0.0f));
	ball->setLinearUpperLimit(btVector3(0.0f, 0.0f,0.0f));
	ball->setAngularLowerLimit(btVector3(-fra, -1.5f, -fra));
	ball->setAngularUpperLimit(btVector3(fra, 1.5f, fra));
	
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
	
	btGeneric6DofConstraint* hinge = PhysicsState::engine->constrainBy6Dof(*ChassisBody, *arm, frmA, frmB, true);
	hinge->setAngularLowerLimit(btVector3(0.0, 0.0, -.2f));
	hinge->setAngularUpperLimit(btVector3(0.0, 0.0, .2f));
	hinge->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
	hinge->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
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
	
	btGeneric6DofConstraint* hinge = PhysicsState::engine->constrainBy6Dof(*ChassisBody, *armBody, frmA, frmB, true);
	
	hinge->setAngularLowerLimit(btVector3(0.f, 0.f, -.2f));
	hinge->setAngularUpperLimit(btVector3(0.f, 0.f, .2f));
	hinge->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
	hinge->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
	
	tmA.setIdentity();
	if(isLeft) tmA.translate(0.f, 0.f, m_profile._steerArmJointZ);
	else tmA.translate(0.f, 0.f, -m_profile._steerArmJointZ);
	frmA = Common::CopyFromMatrix44F(tmA);
	
	tmB.translate(0.f, l * -1.f * cos(ang), 0.f);
	frmB = Common::CopyFromMatrix44F(tmB);
	
	hinge = PhysicsState::engine->constrainBy6Dof(*carrier, *armBody, frmA, frmB, true);
	
	hinge->setAngularLowerLimit(btVector3(-.2f, -.2f, -.2f));
	hinge->setAngularUpperLimit(btVector3(.2f, .2f, .2f));
	hinge->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
	hinge->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
	
	if(isLeft) m_steerJoint[0] = hinge;
	else m_steerJoint[1] = hinge;
	
	return armBody;
}

btRigidBody* Suspension::createDamper(btRigidBody * lowerArm, const Matrix44F & tm)
{
	Matrix44F lowerJntTm;
	
	const float tilt = atan((m_profile._upperWishboneLength - RodRadius * 5.f) / (m_profile._damperY - m_profile._lowerJointY));
	const float l = (m_profile._damperY - m_profile._lowerJointY) / cos(tilt);
	
	lowerJntTm.rotateZ(tilt);
	
	lowerJntTm.translate(Vector3F(RodRadius * -5.f, m_profile._lowerJointY, 0.f));
	
	lowerJntTm *= tm;
	
	Matrix44F damperTm; damperTm.translate(0.f, l * .25f, 0.f);
	damperTm *= lowerJntTm;
	
	btCollisionShape* damperShape = PhysicsState::engine->createCylinderShape(RodRadius, l * .25f, RodRadius);
	btTransform trans = Common::CopyFromMatrix44F(damperTm);
	btRigidBody* damperLowBody = PhysicsState::engine->createRigidBody(damperShape, trans, 1.f);
	damperLowBody->setDamping(0.f, 0.f);
	
	damperTm.setIdentity(); 
	damperTm.translate(0.f, l * .75f, 0.f);
	damperTm *= lowerJntTm;
	trans = Common::CopyFromMatrix44F(damperTm);
	btRigidBody* damperHighBody = PhysicsState::engine->createRigidBody(damperShape, trans, 1.f);
	damperHighBody->setDamping(0.f, 0.f);
	
	damperTm.setIdentity();
	damperTm.rotateZ(- tilt);
	damperTm.translate(0.f, -l * .25f, 0.f);
	btTransform frmA = Common::CopyFromMatrix44F(damperTm);
	
	Matrix44F armTm; armTm.translate(RodRadius * -5.f, 0.f, 0.f);
	armTm.rotateZ(-m_profile._lowerWishboneTilt);
	btTransform frmB = Common::CopyFromMatrix44F(armTm);
	btGeneric6DofConstraint* hinge = PhysicsState::engine->constrainBy6Dof(*damperLowBody, *lowerArm, frmA, frmB, true);
	hinge->setAngularLowerLimit(btVector3(0.f, 0.f, -.5f));
	hinge->setAngularUpperLimit(btVector3(0.f, 0.f, .5f));
	hinge->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
	hinge->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
	
	damperTm.translate(0.f, l * .5f, 0.f);
	frmA = Common::CopyFromMatrix44F(damperTm);
	
	Matrix44F chassisTm; chassisTm.translate(RodRadius * -5.f, m_profile._lowerJointY, 0.f);
	chassisTm.translate(-l * sin(tilt), l * cos(tilt), 0.f);
	chassisTm *= tm;
	chassisTm.setTranslation(chassisTm.getTranslation() - ChassisOrigin);
	frmB = Common::CopyFromMatrix44F(chassisTm);
	hinge = PhysicsState::engine->constrainBy6Dof(*damperHighBody, *ChassisBody, frmA, frmB, true);
	hinge->setAngularLowerLimit(btVector3(0.f, 0.f, -.5f));
	hinge->setAngularUpperLimit(btVector3(0.f, 0.f, .5f));
	hinge->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
	hinge->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
	
	damperTm.setIdentity();
	damperTm.translate(0.f, l * .25f, 0.f);
	frmA = Common::CopyFromMatrix44F(damperTm);
	
	damperTm.translate(0.f, l * -.5f, 0.f);
	frmB = Common::CopyFromMatrix44F(damperTm);
	
	btGeneric6DofSpringConstraint* slid = PhysicsState::engine->constrainBySpring(*damperLowBody, *damperHighBody, frmA, frmB, true);
	slid->setAngularLowerLimit(btVector3(0.f, 0.f, 0.f));
	slid->setAngularUpperLimit(btVector3(0.f, 0.f, 0.f));
	slid->setLinearLowerLimit(btVector3(0.0, -1.5f, 0.0));
	slid->setLinearUpperLimit(btVector3(0.0, 1.5f, 0.0));
	
	slid->enableSpring(1, true);
	slid->setStiffness(1, 600.f);
	slid->setDamping(1, 0.05f);
	slid->setEquilibriumPoint(1, 1.f);
	
	return damperHighBody;
}

btRigidBody* Suspension::createSwayBar(const Matrix44F & tm, btRigidBody * arm, bool isLeft)
{
	const float l = tm.getTranslation().x - ChassisOrigin.x;
	
	float bl = l;
	if(bl < 0.f) bl = -bl;
	
	btCollisionShape* barShape = PhysicsState::engine->createCylinderShape(RodRadius, bl* .5f - 5.f * RodRadius, RodRadius);
	
	Matrix44F barTm; 
	barTm.rotateZ(PI * .5f);
	barTm *= tm;
	
	float xoff = -l * .5f - 5.f * RodRadius;
	if(!isLeft) xoff = -l * .5f + 5.f * RodRadius;
	
	const float zoff = m_profile._lowerWishboneLength * sin(m_profile._lowerWishboneAngle[1]);
	
	barTm.translate(xoff, m_profile._lowerJointY, -zoff);
	
	const Vector3F jnt = barTm.getTranslation();

	btTransform trans = Common::CopyFromMatrix44F(barTm);
	btRigidBody* barBody = PhysicsState::engine->createRigidBody(barShape, trans, 1.f);
	barBody->setDamping(0.f, 0.f);
	
	barTm.setIdentity();
	barTm.translate(0.f, bl* .5f - 5.f * RodRadius -bl, 0.f);
	if(isLeft) barTm.translate(0.f, 0.f, zoff);
	else barTm.translate(0.f, 0.f, -zoff);
	
	btTransform frmBar = Common::CopyFromMatrix44F(barTm);
	btTransform frmArm = Common::CopyFromMatrix44F(Matrix44F());
	
	btGeneric6DofConstraint* ball = PhysicsState::engine->constrainBy6Dof(*barBody, *arm, frmBar, frmArm, true);
	ball->setLinearLowerLimit(btVector3(0.f, 0.f, 0.f));
	ball->setLinearUpperLimit(btVector3(0.f, 0.f, 0.f));
	ball->setAngularLowerLimit(btVector3(-PI, -PI, -PI));
	ball->setAngularUpperLimit(btVector3(PI, PI, PI));
	
	barTm.setIdentity();
	
	Matrix44F chassisTm;
	chassisTm.rotateZ(PI * .5f);
	if(!isLeft) {
		Matrix44F flipX; flipX.rotateY(PI);
		chassisTm *= flipX;
	}
	
	chassisTm.setTranslation(jnt - ChassisOrigin);
	
	frmBar = Common::CopyFromMatrix44F(barTm);
	btTransform frmChassis = Common::CopyFromMatrix44F(chassisTm);
	
	ball = PhysicsState::engine->constrainBy6Dof(*barBody, *ChassisBody, frmBar, frmChassis, true);
	ball->setLinearLowerLimit(btVector3(0.f, 0.f, 0.f));
	ball->setLinearUpperLimit(btVector3(0.f, 0.f, 0.f));
	ball->setAngularLowerLimit(btVector3(0.f, -PI, 0.f));
	ball->setAngularUpperLimit(btVector3(0.f, PI, 0.f));
	
	return barBody;
}

void Suspension::connectSwayBar(const Matrix44F & tm, btRigidBody * bar)
{
	const float l = tm.getTranslation().x - ChassisOrigin.x;
	
	float bl = l;
	if(bl < 0.f) bl = -bl;
	bl *= .5f;
	bl -= 5.f * RodRadius;
	
	btTransform frmA; frmA.setIdentity(); frmA.getOrigin()[1] = bl;
	
	Matrix44F bTm; bTm.rotateX(PI);
	bTm.translate(0.f, bl, 0.f);
	btTransform frmB = Common::CopyFromMatrix44F(bTm);
	
	btGeneric6DofSpringConstraint* ball = PhysicsState::engine->constrainBySpring(*m_swayBarLeft, *bar, frmA, frmB, true);
	ball->setLinearLowerLimit(btVector3(0.f, -1.f, 0.f));
	ball->setLinearUpperLimit(btVector3(0.f, 1.f, 0.f));
	ball->setAngularLowerLimit(btVector3(0.f, -PI, 0.f));
	ball->setAngularUpperLimit(btVector3(0.f, PI, 0.f));
	
	ball->enableSpring(4, true);
	ball->setStiffness(4, 600.f);
	ball->setDamping(4, 0.05f);
	ball->setEquilibriumPoint(4, 0.f);
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

void Suspension::powerDrive(const float & ang, const float & wheelSpan, const Vector3F & targetVelocity, const float & wheelR, bool goForward)
{
	const float speed = targetVelocity.length();
	if(speed == 0.f) return applyBrake(true);
	else applyBrake(false);
	
	if(!isPowered()) return;
	
	if(ang < -0.001f || ang > 0.001f) {
		const float ds = wheelSpan * .5f / (speed / tan(ang));
		// std::cout<<"lft/rgt "<<ds<<" / "<<-ds<<"\n";
		limitDrive(0, speed, wheelR, ds, goForward);
		limitDrive(1, speed, wheelR, -ds, goForward);
	}
	else {
		limitDrive(0, speed, wheelR, 0.f, goForward);
		limitDrive(1, speed, wheelR, 0.f, goForward);
	}
}

float Suspension::limitDrive(const int & i, const float & targetSpeed, const float & r, const float & differential, bool goForward)
{
	float wheelSpeed = wheelVelocity(i).length();
	float diff = targetSpeed - wheelSpeed;
	
	//float low = (80.f - wheelSpeed) / 80.f;
	//if(low < 0.f) low = 0.f;
	
	const float lmt = r * SPEEDLIMIT;// * (1.f - .5f * low);
	if(diff > lmt) diff = lmt;
	else if(diff < -lmt) diff = -lmt;
	
	wheelSpeed += diff;
	wheelSpeed *= 1.f + differential;
	std::cout<<"limit ["<<i<<"] "<<wheelSpeed;
	float rps = wheelSpeed / r;
	if(!goForward) rps = -rps;
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
	m_driveJoint[i]->getRotationalLimitMotor(0)->m_maxMotorForce = 33.f;
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