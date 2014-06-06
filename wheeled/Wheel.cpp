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

const float Wheel::radius() const { return m_profile._radiusMajor; }

void Wheel::createShape()
{
	
}

btRigidBody* Wheel::create(const Matrix44F & tm, bool isLeft)
{
    m_shape = m_tire.create(m_profile._radiusMajor, m_profile._radiusMinor, m_profile._width, isLeft);
    
	btTransform trans = Common::CopyFromMatrix44F(tm);
	btRigidBody* wheelBody = PhysicsState::engine->createRigidBody(m_shape, trans, 3.f);
	wheelBody->setDamping(0.f, 0.f);
	wheelBody->setFriction(19.99f);
	wheelBody->setActivationState(DISABLE_DEACTIVATION);
	m_tire.attachPad(wheelBody, tm, m_profile._radiusMajor, m_profile._radiusMinor, isLeft);
	return wheelBody;
}

}