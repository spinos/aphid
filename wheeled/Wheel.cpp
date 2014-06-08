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

btRigidBody* Wheel::create(const Matrix44F & tm, bool isLeft)
{
    m_body = m_tire.create(m_profile._radiusMajor, m_profile._radiusMinor, m_profile._width, m_profile._mass, tm, isLeft);
	return m_body;
}

btRigidBody* Wheel::body() { return m_body; }

const Vector3F Wheel::velocity() const 
{
	const btVector3 vel = m_body->getLinearVelocity(); 
	return Vector3F(vel[0], vel[1], vel[2]);
}

const Vector3F Wheel::angularVelocity() const
{
	const btVector3 vel = m_body->getAngularVelocity(); 
	return Vector3F(vel[0], vel[1], vel[2]);
}

const btTransform Wheel::tm() const 
{
	return m_body->getWorldTransform(); 
}

}