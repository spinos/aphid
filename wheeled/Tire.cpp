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
#define NUMGRIDRAD 91
	
Tire::Tire() {}
Tire::~Tire() {}
btCollisionShape* Tire::create(const float & radiusMajor, const float & radiusMinor, const float & width)
{
	const float sy = (radiusMajor - radiusMinor) * PI / NUMGRIDRAD;
	btCompoundShape* wheelShape = new btCompoundShape();
	m_padShape = PhysicsState::engine->createBoxShape(width * .5f, sy, radiusMinor * .5);
	const float delta = PI * 2.f / (float)NUMGRIDRAD;
	int i;
	Matrix44F rot;
	for(i = 0; i < NUMGRIDRAD; i++) {
		Matrix44F ctm;
		ctm.translate(0.f, 0.f, radiusMajor - radiusMinor * .5f);
		ctm *= rot;
		btTransform frm = Common::CopyFromMatrix44F(ctm);
		wheelShape->addChildShape(frm, m_padShape);
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