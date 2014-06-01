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
float Suspension::RodRadius = .29f;
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
	btRigidBody* upperArm = createUpperWishbone(tm.getTranslation(), isLeft);
	btRigidBody* lowerArm = createLowerWishbone(tm.getTranslation(), isLeft);
	
	btTransform frmCarrier; frmCarrier.setIdentity();
	frmCarrier.getOrigin()[1] = m_profile._upperJointY;
	
	btTransform frmArm; frmArm.setIdentity();
	
	btGeneric6DofConstraint* ball = PhysicsState::engine->constrainBy6Dof(*carrier, *upperArm, frmCarrier, frmArm, true);
	frmCarrier.getOrigin()[1] = m_profile._lowerJointY;
	ball = PhysicsState::engine->constrainBy6Dof(*carrier, *lowerArm, frmCarrier, frmArm, true);
	
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
	btRigidBody* carrierBody = PhysicsState::engine->createRigidBody(carrierShape, trans, 1.f);
	carrierBody->setDamping(0.f, 0.f);
	
	return carrierBody;
}

btRigidBody* Suspension::createUpperWishbone(const Vector3F & pos, bool isLeft)
{
	Matrix44F btm;
	if(!isLeft) btm.rotateY(PI);
	
	btm.rotateZ(m_profile._upperWishboneTilt);
	
	btm.translate(Vector3F(0.f, m_profile._upperJointY, 0.f));
	btm.translate(pos);
	
	btCompoundShape* wishboneShape = createWishboneShape(true, isLeft);
	
	btTransform trans = Common::CopyFromMatrix44F(btm);
	btRigidBody* wishboneBody = PhysicsState::engine->createRigidBody(wishboneShape, trans, 1.f);
	wishboneBody->setDamping(0.f, 0.f);
	
	connectArm(wishboneBody, pos, true, isLeft, true);
	connectArm(wishboneBody, pos, true, isLeft, false);
	
	return wishboneBody;
}

btRigidBody* Suspension::createLowerWishbone(const Vector3F & pos, bool isLeft)
{
	Matrix44F btm;
	if(!isLeft) btm.rotateY(PI);
	
	btm.rotateZ(m_profile._lowerWishboneTilt);
	
	btm.translate(Vector3F(0.f, m_profile._lowerJointY, 0.f));
	btm.translate(pos);
	
	btCompoundShape* wishboneShape = createWishboneShape(false, isLeft);
	
	btTransform trans = Common::CopyFromMatrix44F(btm);
	btRigidBody* wishboneBody = PhysicsState::engine->createRigidBody(wishboneShape, trans, 1.f);
	wishboneBody->setDamping(0.f, 0.f);
	
	connectArm(wishboneBody, pos, false, isLeft, true);
	connectArm(wishboneBody, pos, false, isLeft, false);
	
	return wishboneBody;
}

void Suspension::connectArm(btRigidBody* arm, const Vector3F & pos, bool isUpper, bool isLeft, bool isFront)
{
	Matrix44F localBone = wishboneHingTMLocal(isUpper, isLeft, isFront);
	btTransform frmB = Common::CopyFromMatrix44F(localBone);
	
	Matrix44F hingeTM = localBone;
	if(!isLeft) hingeTM.rotateY(PI);
	
	if(isUpper) {
		if(isLeft) hingeTM.rotateZ(-m_profile._upperWishboneTilt);
		else hingeTM.rotateZ(m_profile._upperWishboneTilt);
		hingeTM.translate(Vector3F(0.f, m_profile._upperJointY, 0.f));
	}
	else {
		if(isLeft) hingeTM.rotateZ(-m_profile._lowerWishboneTilt);
		else hingeTM.rotateZ(m_profile._lowerWishboneTilt);
		hingeTM.translate(Vector3F(0.f, m_profile._lowerJointY, 0.f));
	}
	
	hingeTM.translate(pos);
	
	Matrix44F localChas;
	localChas.setTranslation(hingeTM.getTranslation() - ChassisOrigin);
		
	btTransform frmA = Common::CopyFromMatrix44F(localChas);
	
	btGeneric6DofSpringConstraint* hinge = PhysicsState::engine->constrainBySpring(*ChassisBody, *arm, frmA, frmB, true);
	//hinge->setAngularLowerLimit(btVector3(-PI, 0, -PI));
	//hinge->setAngularUpperLimit(btVector3(PI, 0, PI));
	//hinge->setLinearLowerLimit(btVector3(0.0, 0.0, 0.0));
	//hinge->setLinearUpperLimit(btVector3(0.0, 0.0, 0.0));
}

btCompoundShape* Suspension::createWishboneShape(bool isUpper, bool isLeft)
{
	btCompoundShape* shape = new btCompoundShape();
	
	float l, ang;
	wishboneLA(isUpper, isLeft, true, l, ang);
		
	Matrix44F tm;
	tm.rotateY(ang);
	tm.rotateZ(PI * .5f);
	tm.setTranslation(Vector3F(-l * .5f * cos(ang), 0.f, -l * .5f * sin(ang)));
	
	btTransform childT = Common::CopyFromMatrix44F(tm);
	
	btCollisionShape* armShape = PhysicsState::engine->createCylinderShape(RodRadius, l * .5f, RodRadius);
	
	shape->addChildShape(childT, armShape);
	
	wishboneLA(isUpper, isLeft, false, l, ang);
	
	tm.setIdentity();
	tm.rotateY(ang);
	tm.rotateZ(PI * .5f);
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

const float Suspension::wheelHubX() const { return m_profile._wheelHubX; }

}