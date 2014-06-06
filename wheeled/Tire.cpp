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
#define NUMGRIDRAD 30
#define PADROLL .6f
Tire::Tire() {}
Tire::~Tire() {}
btCollisionShape* Tire::create(const float & radiusMajor, const float & radiusMinor, const float & width, bool isLeft)
{
	const float bent = .06f;
	const float bent2 = .15f;
	const float bent15 = .072f;
	const float hw = .5f * width; m_hw = hw;
	const float sy = (radiusMajor - radiusMinor) * PI / NUMGRIDRAD;
	m_padShape = PhysicsState::engine->createBoxShape(hw * .4f, sy * .5f, radiusMinor * .5);
	
	btCompoundShape* wheelShape = new btCompoundShape();

	float d = -1.f;
	if(isLeft) d= 1.f;
	Matrix44F tm[2];
	tm[0].rotateZ(PADROLL * d);
	tm[0].translate(-.5f * hw, 0.f, radiusMajor - radiusMinor * 1.5f);
	
	tm[1].rotateZ(-PADROLL * d);
	tm[1].translate(.5f * hw, 0.f, radiusMajor - radiusMinor * 1.5f);

	const float delta = PI * 2.f / (float)NUMGRIDRAD;
	int i, j;
	Matrix44F rot;
	for(i = 0; i < NUMGRIDRAD; i++) {
		for(j = 0; j < 2; j++) {
			Matrix44F ctm = tm[j];
			ctm *= rot;
			btTransform frm = Common::CopyFromMatrix44F(ctm);
			wheelShape->addChildShape(frm, m_padShape);
		}
		rot.rotateX(delta);
	}
	
	PhysicsState::engine->addCollisionShape(wheelShape);
	
	return wheelShape;
}

void Tire::attachPad(btRigidBody* wheelBody, const Matrix44F & origin, const float & radiusMajor, const float & radiusMinor, bool isLeft)
{
    const float delta = PI * 2.f / (float)NUMGRIDRAD;
	int i, j;
	
	float d = -1.f;
	if(isLeft) d= 1.f;
	
	Matrix44F tm[2];
	tm[0].rotateZ(PADROLL * d);
	tm[0].translate(-.5f * m_hw, 0.f, radiusMajor - radiusMinor * .5f);
	
	tm[1].rotateZ(-PADROLL * d);
	tm[1].translate(.5f * m_hw, 0.f, radiusMajor - radiusMinor * .5f);

	Matrix44F rot;
	for(i = 0; i < NUMGRIDRAD; i++) {
	    for(j = 0; j < 2; j++) {
		Matrix44F ctm = tm[j];
		ctm *= rot;
		
		btTransform frameInA = Common::CopyFromMatrix44F(ctm);
		
		ctm *= origin;
		
		btTransform frm = Common::CopyFromMatrix44F(ctm);
		btRigidBody* padBody = PhysicsState::engine->createRigidBody(m_padShape, frm, .25f);
		padBody->setFriction(19.99f);
		padBody->setDamping(0.f, 0.f);
		
		btTransform frameInB; frameInB.setIdentity();
		btGeneric6DofSpringConstraint* spring = PhysicsState::engine->constrainBySpring(*wheelBody, *padBody, frameInA, frameInB, true);
		
		spring->setLinearUpperLimit(btVector3(0., 0., -.5f * radiusMinor));
		spring->setLinearLowerLimit(btVector3(0., 0., .5f * radiusMinor));
		spring->setAngularLowerLimit(btVector3( -0.1f, -0.1f, 0.f));
	    spring->setAngularUpperLimit(btVector3(0.1f, 0.1f, 0.f));
	    spring->enableSpring(2, true);
	    spring->setStiffness(2, 1100.f);
	    spring->setDamping(2, 0.001f);
		spring->setEquilibriumPoint(2, 0.0f);
		
		spring->enableSpring(3, true);
	    spring->setStiffness(3, 1100.f);
	    spring->setDamping(3, 0.001f);
		spring->setEquilibriumPoint(3, 0.0f);
		
		spring->enableSpring(4, true);
	    spring->setStiffness(4, 1100.f);
	    spring->setDamping(4, 0.001f);
		spring->setEquilibriumPoint(4, 0.0f);
		
		}
		rot.rotateX(delta);
	}
}
}