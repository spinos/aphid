/*
 *  Damper.cpp
 *  wheeled
 *
 *  Created by jian zhang on 6/6/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */

#include "Damper.h"
#include <DynamicsSolver.h>
#include <PhysicsState.h>
namespace caterpillar {
Damper::Damper(btRigidBody * lower, btRigidBody * upper, const float & l) 
{
	m_body[0] = lower;
	m_body[1] = upper;
	m_range = l;
	
	Matrix44F damperTm;
	damperTm.translate(0.f, l, 0.f);
	btTransform frmA = Common::CopyFromMatrix44F(damperTm);
	
	damperTm.translate(0.f, l * -2.f - m_range, 0.f);
	btTransform frmB = Common::CopyFromMatrix44F(damperTm);
	
	btGeneric6DofSpringConstraint* slid = PhysicsState::engine->constrainBySpring(*lower, *upper, frmA, frmB, true);
	slid->setAngularLowerLimit(btVector3(0.f, 0.f, 0.f));
	slid->setAngularUpperLimit(btVector3(0.f, 0.f, 0.f));
	slid->setLinearLowerLimit(btVector3(0.0, -m_range, 0.0));
	slid->setLinearUpperLimit(btVector3(0.0, m_range, 0.0));
	
	slid->enableSpring(1, true);
	slid->setStiffness(1, 500.f);
	slid->setDamping(1, 0.05f);
	slid->setEquilibriumPoint(1, 0.f);
	m_slid = slid;
}

Damper::~Damper() {}

void Damper::update() 
{
	if(isCompressing())
		m_slid->setDamping(1, 0.1f);
	else
		m_slid->setDamping(1, 0.01f);
}

const bool Damper::isCompressing() const
{
	return relativeSpeed() > 0.f;
}

const float Damper::relativeSpeed() const
{
	const btTransform tm = m_body[1]->getWorldTransform();
	Matrix44F space = Common::CopyFromBtTransform(tm);
	space.inverse();
	
	const btVector3 vel = m_body[0]->getLinearVelocity(); 
	Vector3F v(vel[0], vel[1], vel[2]);
	v = space.transformAsNormal(v);
	
	return v.y;
}

}