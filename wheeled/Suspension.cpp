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

void Suspension::create(const Vector3F & pos, bool isLeft) 
{
	Matrix44F tm;
	if(!isLeft) tm.rotateY(PI);
	
	if(isLeft)
		tm.setTranslation(-m_profile._wheelHubX, 0.f, 0.f);
	else
		tm.setTranslation(m_profile._wheelHubX, 0.f, 0.f);
	
	tm.translate(pos);

	createCarrier(tm, isLeft);
	createUpperWishbone(tm, isLeft);
	createLowerWishbone(tm, isLeft);
}

void Suspension::createCarrier(const Matrix44F & tm, bool isLeft)
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
}

void Suspension::createUpperWishbone(const Matrix44F & tm, bool isLeft)
{
	Matrix44F btm;
	btm.rotateZ(m_profile._upperWishboneTilt);
	btm.translate(Vector3F(0.f, m_profile._upperJointY, 0.f));
	btm.transformBy(tm);
	
	btCompoundShape* wishboneShape = createWishboneShape(true, isLeft);
	
	btTransform trans = Common::CopyFromMatrix44F(btm);
	btRigidBody* wishboneBody = PhysicsState::engine->createRigidBody(wishboneShape, trans, 1.f);
	wishboneBody->setDamping(0.f, 0.f);
}

void Suspension::createLowerWishbone(const Matrix44F & tm, bool isLeft)
{
	Matrix44F btm;
	btm.rotateZ(m_profile._lowerWishboneTilt);
	btm.translate(Vector3F(0.f, m_profile._lowerJointY, 0.f));
	btm.transformBy(tm);
	
	btCompoundShape* wishboneShape = createWishboneShape(false, isLeft);
	
	btTransform trans = Common::CopyFromMatrix44F(btm);
	btRigidBody* wishboneBody = PhysicsState::engine->createRigidBody(wishboneShape, trans, 1.f);
	wishboneBody->setDamping(0.f, 0.f);
}

btCompoundShape* Suspension::createWishboneShape(bool isUpper, bool isLeft)
{
	btCompoundShape* shape = new btCompoundShape();
	
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
		angA = -angA;
		angB = -angB;
	}
	
	Matrix44F tm;
	tm.rotateY(angA);
	tm.rotateZ(PI * .5f);
	tm.setTranslation(Vector3F(-lA * .5f * cos(angA), 0.f, -lA * .5f * sin(angA)));
	
	btTransform childT = Common::CopyFromMatrix44F(tm);
	
	btCollisionShape* armShape = PhysicsState::engine->createCylinderShape(RodRadius, lA * .5f, RodRadius);
	
	shape->addChildShape(childT, armShape);
	
	tm.setIdentity();
	tm.rotateY(angB);
	tm.rotateZ(PI * .5f);
	tm.setTranslation(Vector3F(-lB * .5f * cos(angB), 0.f, -lB * .5f * sin(angB)));
	
	childT = Common::CopyFromMatrix44F(tm);
	
	armShape = PhysicsState::engine->createCylinderShape(RodRadius, lB * .5f, RodRadius);
	
	shape->addChildShape(childT, armShape);
	
	return shape;
}

}