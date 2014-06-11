/*
 *  Tire.cpp
 *  wheeled
 *
 *  Created by jian zhang on 6/1/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Tire.h"
#include "PhysicsState.h"
#include <DynamicsSolver.h>
namespace caterpillar {
#define PADROLL 0.f
Tire::Tire() {}
Tire::~Tire() {}
btRigidBody* Tire::create(const float & radiusMajor, const float & radiusMinor, const float & width, const float & mass, const Matrix44F & tm, bool isLeft)
{
	const float hw = .5f * width; m_hw = hw;
	const float sy = (radiusMajor - radiusMinor) * PI / NUMGRIDRAD;
	btCollisionShape * padShape = PhysicsState::engine->createBoxShape(hw, sy * .8f, radiusMinor * .5);
	// btCollisionShape * padShape = PhysicsState::engine->createCylinderShape(sy * .9f, hw, sy * .9f);
	
	btCompoundShape* wheelShape = new btCompoundShape();

	float d = -1.f;
	if(isLeft) d= 1.f;
	Matrix44F ptm[2];
	ptm[0].rotateZ(PADROLL * d);
	// ptm[0].rotateZ(PI * .5f);
	ptm[0].translate(-0.f * hw, 0.f, radiusMajor - radiusMinor * 1.5f);
	
	ptm[1].rotateZ(-PADROLL * d);
	ptm[1].translate(.5f * hw, 0.f, radiusMajor - radiusMinor * 1.5f);

	const float delta = PI * 2.f / (float)NUMGRIDRAD;
	int i;
	Matrix44F rot;
	for(i = 0; i < NUMGRIDRAD; i++) {
		Matrix44F ctm = ptm[0];
		ctm *= rot;
		btTransform frm = Common::CopyFromMatrix44F(ctm);
		wheelShape->addChildShape(frm, padShape);
		rot.rotateX(delta);
	}
	
	PhysicsState::engine->addCollisionShape(wheelShape);
	
	btTransform trans = Common::CopyFromMatrix44F(tm);
	btRigidBody* wheelBody = PhysicsState::engine->createRigidBody(wheelShape, trans, mass);
	wheelBody->setDamping(0.f, 0.f);
	wheelBody->setFriction(1.99f);
	wheelBody->setActivationState(DISABLE_DEACTIVATION);
	
	attachPad(wheelBody, padShape, tm, radiusMajor, radiusMinor, isLeft);
	
	return wheelBody;
}

void Tire::attachPad(btRigidBody* wheelBody, btCollisionShape * padShape, const Matrix44F & origin, const float & radiusMajor, const float & radiusMinor, bool isLeft)
{
    const float delta = PI * 2.f / (float)NUMGRIDRAD;
	int i;
	
	float d = -1.f;
	if(isLeft) d= 1.f;
	
	Matrix44F tm[2];
	tm[0].rotateZ(PADROLL * d);
	// tm[0].rotateZ(PI * .5f);
	tm[0].translate(-0.f * m_hw, 0.f, radiusMajor - radiusMinor * .5f);
	
	tm[1].rotateZ(-PADROLL * d);
	tm[1].translate(.5f * m_hw, 0.f, radiusMajor - radiusMinor * .5f);

	Matrix44F rot;
	for(i = 0; i < NUMGRIDRAD; i++) {
	    Matrix44F ctm = tm[0];
		ctm *= rot;
		
		btTransform frameInA = Common::CopyFromMatrix44F(ctm);
		
		ctm *= origin;
		
		btTransform frm = Common::CopyFromMatrix44F(ctm);
		btRigidBody* padBody = PhysicsState::engine->createRigidBody(padShape, frm, .1f);
		padBody->setFriction(3.99f);
		padBody->setDamping(0.f, 0.f);
		m_bd[i] = padBody;
		
		btTransform frameInB; frameInB.setIdentity();
		btGeneric6DofSpringConstraint* spring = PhysicsState::engine->constrainBySpring(*wheelBody, *padBody, frameInA, frameInB, true);
		
		spring->setLinearUpperLimit(btVector3(0., 0., -.25f * radiusMinor));
		spring->setLinearLowerLimit(btVector3(0., 0., .25f * radiusMinor));
		spring->setAngularLowerLimit(btVector3(-0.05f, -0.05f, 0.f));
	    spring->setAngularUpperLimit(btVector3(0.05f, 0.05f, 0.f));
	    spring->enableSpring(2, true);
	    spring->setStiffness(2, 1900.f);
	    spring->setDamping(2, 0.001f);
		spring->setEquilibriumPoint(2, 0.0f);
		
		spring->enableSpring(3, true);
		spring->setStiffness(3, 1900.f);
		spring->setDamping(3, 0.001f);
		spring->setEquilibriumPoint(3, 0.0f);
		
		spring->enableSpring(4, true);
	    spring->setStiffness(4, 1900.f);
	    spring->setDamping(4, 0.001f);
		spring->setEquilibriumPoint(4, 0.0f);
		
		rot.rotateX(delta);
	}
}

void Tire::setFriction(const float & x)
{
    for(int i = 0; i < NUMGRIDRAD; i++) m_bd[i]->setFriction(x);;
}

}