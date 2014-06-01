/*
 *  Wheel.cpp
 *  wheeled
 *
 *  Created by jian zhang on 5/31/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Wheel.h"
#include <DynamicsSolver.h>
#include <PhysicsState.h>
#include <Common.h>
namespace caterpillar {
Wheel::Wheel() {}
Wheel::~Wheel() {}
void Wheel::setProfile(const Profile & info) { m_profile = info; }

const float Wheel::width() const { return m_profile._width; }

void Wheel::create(const Matrix44F & tm) 
{
	btCollisionShape* wheelShape = PhysicsState::engine->createCylinderShape(m_profile._radiusMajor, m_profile._width * .5f, m_profile._radiusMajor);
	btTransform trans = Common::CopyFromMatrix44F(tm);
	btRigidBody* wheelBody = PhysicsState::engine->createRigidBody(wheelShape, trans, 1.f);
	wheelBody->setDamping(0.f, 0.f);
	wheelBody->setFriction(.732f);
	
}

}