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
#define NUMGRIDRAD 60
	
Tire::Tire() {}
Tire::~Tire() {}
btCollisionShape* Tire::create(const float & radiusMajor, const float & radiusMinor, const float & width)
{
	const float bent = .06f;
	const float bent2 = .15f;
	const float bent15 = .072f;
	const float hw = .5f * width;
	const float sy = (radiusMajor - radiusMinor) * PI / NUMGRIDRAD;
	btCompoundShape* wheelShape = new btCompoundShape();
	m_padShape = PhysicsState::engine->createBoxShape(hw * .23f, sy * .8f, radiusMinor * .5);
	Matrix44F tm[4];
	tm[0].rotateZ(.5f);
	tm[0].rotateY(-bent2);
	tm[0].translate(-.75f * hw, 0.f, radiusMajor - radiusMinor * .5f - .75f * hw * sin(bent15));
	
	tm[1].rotateZ(-.5f);
	tm[1].rotateY(-bent);
	tm[1].translate(-.25f * hw, 0.f, radiusMajor - radiusMinor * .5f);
	
	tm[2].rotateZ(.6f);
	tm[2].rotateY(bent);
	tm[2].translate(.25f * hw, 0.f, radiusMajor - radiusMinor * .5f);
	
	tm[3].rotateZ(-.7f);
	tm[3].rotateY(bent2);
	tm[3].translate(.75f * hw, 0.f, radiusMajor - radiusMinor * .5f - .75f * hw * sin(bent15));
	
	
	const float delta = PI * 2.f / (float)NUMGRIDRAD;
	int i, j;
	Matrix44F rot;
	for(i = 0; i < NUMGRIDRAD; i++) {
		for(j = 0; j < 4; j++) {
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
	int i;

	Matrix44F rot;
	for(i = 0; i < NUMGRIDRAD; i++) {
		Matrix44F ctm;
		ctm.translate(0.f, 0.f, radiusMajor);
		ctm *= rot;
		
		btTransform frameInA = Common::CopyFromMatrix44F(ctm);
		
		ctm *= origin;
		
		btTransform frm = Common::CopyFromMatrix44F(ctm);
		btRigidBody* padBody = PhysicsState::engine->createRigidBody(m_padShape, frm, .25f);
		padBody->setFriction(.9f);
		padBody->setDamping(0.f, 0.f);
		btTransform frameInB; frameInB.setIdentity();
		btGeneric6DofSpringConstraint* spring = PhysicsState::engine->constrainBySpring(*wheelBody, *padBody, frameInA, frameInB, true);
		
		spring->setLinearUpperLimit(btVector3(0., 0., -radiusMinor*.25f));
		spring->setLinearLowerLimit(btVector3(0., 0., radiusMinor*.25f));
		spring->setAngularLowerLimit(btVector3(0.f, 0.f, 0.f));
	    spring->setAngularUpperLimit(btVector3(0.f, 0.f, 0.f));
	    spring->enableSpring(2, true);
	    spring->setStiffness(2, 1200.f);
	    spring->setDamping(2, 0.01f);
		spring->setEquilibriumPoint(2, 0.01f);
		rot.rotateX(delta);
	}
}
}