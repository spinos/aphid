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
	Matrix44F tm; tm.rotateY(PI);
	
	if(isLeft) {
		tm.rotateZ(PI * -.5f);
		tm.setTranslation(-m_profile._wheelHubX * .5f, 0.f, 0.f);
	}
	else {
		tm.rotateZ(PI * .5f);
		tm.setTranslation(m_profile._wheelHubX * .5f, 0.f, 0.f);
	}
	
	tm.translate(pos);

	btCollisionShape* carrierShape = PhysicsState::engine->createCylinderShape(.5f, m_profile._wheelHubX * .5f, .5f);
	btTransform trans = Common::CopyFromMatrix44F(tm);
	btRigidBody* carrierBody = PhysicsState::engine->createRigidBody(carrierShape, trans, 1.f);
	carrierBody->setDamping(0.f, 0.f);
}

}