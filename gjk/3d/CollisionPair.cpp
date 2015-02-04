/*
 *  CollisionPair.cpp
 *  proof
 *
 *  Created by jian zhang on 2/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "CollisionPair.h"

CollisionPair::CollisionPair(RigidBody * a, RigidBody * b)
{
	m_A = a; m_B = b;
}

void CollisionPair::continuousCollisionDetection(const float & h)
{
	ContinuousCollisionContext &io = m_ccd;
	io.positionA = m_A->position;
	io.positionB = m_B->position;
	io.orientationA = m_A->orientation;
	io.orientationB = m_B->orientation;
	io.linearVelocityA = m_A->linearVelocity * h;
	io.linearVelocityB = m_B->linearVelocity * h;
	io.angularVelocityA = m_A->angularVelocity * h;
	io.angularVelocityB = m_B->angularVelocity * h;
	m_gjk.timeOfImpact(*m_A->shape, *m_B->shape, &m_ccd);
}

void CollisionPair::progressToImpactPostion(const float & h)
{
// only B at impact position
	m_ccd.positionB = m_B->position.progress(m_B->linearVelocity, m_ccd.TOI * h);
	m_ccd.orientationB = m_B->orientation.progress(m_B->angularVelocity, m_ccd.TOI * h);
}

void CollisionPair::detectAtImpactPosition(const float & h)
{
	m_ccd.linearVelocityB = m_B->projectedLinearVelocity * h;
	m_ccd.angularVelocityB = m_B->projectedAngularVelocity * h;
	m_gjk.timeOfImpact(*m_A->shape, *m_B->shape, &m_ccd);
}

#ifdef DBG_DRAW
void CollisionPair::setDrawer(KdTreeDrawer * d)
{ m_gjk.m_dbgDrawer = d; }
#endif

const char CollisionPair::hasContact() const
{ return m_ccd.hasContact; }

const float CollisionPair::TOI() const
{ return m_ccd.TOI; }

void CollisionPair::computeLinearImpulse(float & MinvJa, float & MinvJb, Vector3F & N)
{
// http://www.cs.uu.nl/docs/vakken/mgp/lectures/lecture%207%20Collision%20Resolution.pdf	
	Vector3F Vrel = relativeVelocity();
	
	const float massinv = m_B->shape->linearMassM.x;
	
	float MinvJ = Vrel.dot(m_ccd.contactNormal) * massinv;
	
	MinvJa = -(1.f + .5f) * MinvJ;
	MinvJb = (1.f + .67f) * MinvJ;
	if(m_ccd.penetrateDepth > 0.f) {
		// std::cout<<" penetrate d add relative velocity"<<m_ccd.penetrateDepth;
		MinvJb += m_ccd.penetrateDepth * 48.f;
	}
	std::cout<<" MinvJb "<<MinvJb;
	N = m_ccd.contactNormal;
}

const Vector3F CollisionPair::relativeVelocity() const
{
	return m_A->projectedLinearVelocity - m_B->projectedLinearVelocity;
}
