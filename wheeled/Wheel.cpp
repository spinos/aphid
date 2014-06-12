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
Wheel::Wheel() 
{
    m_friction = 2.99f;
    m_rigidBodyId = 0;
}

Wheel::~Wheel() {}
void Wheel::setProfile(const Profile & info) { m_profile = info; }

const float Wheel::width() const { return m_profile._width; }

const float Wheel::radius() const { return m_profile._radiusMajor; }

btRigidBody* Wheel::create(const Matrix44F & tm, bool isLeft)
{
    m_rigidBodyId = PhysicsState::engine->numCollisionObjects();
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

float Wheel::computeFriction(const float & slipAngle)
{
    float ang = slipAngle;
    if(ang < 0.f) ang = - ang;
    
    float alpha = (ang - 0.03f) * 1.1f;
    if(alpha < 0.f) alpha = 0.f;
    if(alpha > 1.57f) alpha = 1.57f;
    alpha = sin(alpha);
    m_friction = 2.99f - 2.5f * alpha;
    m_tire.setFriction(m_friction);
    return m_friction;
}

const float Wheel::friction() const { return m_friction; }

const int Wheel::rigidBodyId() const {return m_rigidBodyId;}

}