/*
 *  CollisionPair.cpp
 *  proof
 *
 *  Created by jian zhang on 2/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "CollisionPair.h"
#ifdef DBG_DRAW
#include <KdTreeDrawer.h>
#endif
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
	
	float MinvJ = Vrel.dot(m_ccd.contactNormal);
	
	MinvJa = -(1.f + m_A->Crestitution) * MinvJ;
	MinvJb = (1.f + m_B->Crestitution) * MinvJ;
	if(m_ccd.penetrateDepth > 0.f) {
		// std::cout<<" penetrate d add relative velocity"<<m_ccd.penetrateDepth;
		MinvJb += m_ccd.penetrateDepth * 48.f;
	}
	// std::cout<<" MinvJb "<<MinvJb;
	N = m_ccd.contactNormal;
}

const Vector3F CollisionPair::relativeVelocity() const
{
	return m_A->projectedLinearVelocity - m_B->projectedLinearVelocity;
}

void CollisionPair::computeAngularImpulse(Vector3F & IinvJa, float & MinvJa, Vector3F & IinvJb, float & MinvJb)
{
	const Vector3F Vrel = relativeVelocity();
	
	Matrix33F R; 
	R.set(m_ccd.orientationB);
	R.inverse();
	//Vector3F na = R.transform(m_ccd.contactNormal); na.normalize();
	//R.set(m_ccd.orientationB); R.inverse();
	Vector3F nb = R.transform(m_ccd.contactNormal + Vrel); nb.normalize();
	// Vector3F nb = m_ccd.contactNormal;
	const Vector3F rb = m_ccd.contactPointB.reversed();// R.transform(m_ccd.contactPointB.reversed());//
	
	nb.verbose(" nb ");
	rb.verbose(" rb ");
	rb.cross(nb).verbose(" rb x nb ");
	m_B->shape->angularMassM.transform(rb.cross(nb)).verbose(" massed ");
		
	const float massinv = m_B->shape->linearMassM.x;
	
	IinvJb = rb.cross(nb);//;
	
	float up = (m_A->projectedAngularVelocity - m_B->projectedAngularVelocity).dot(IinvJb)/(m_B->shape->angularMassM.transform(rb.cross(nb)).dot(IinvJb));
	if(up==0.f) up = 1.f;
	const float MinvJ =  (up) / ( IinvJb.cross(rb).dot(nb));

	MinvJb = (1.f + .5f) * MinvJ;
	
#ifdef DBG_DRAW
	m_B->r = m_ccd.contactPointB;
	m_B->J = IinvJb;
	m_B->Jsize = MinvJb;
    KdTreeDrawer * drawer = m_gjk.m_dbgDrawer;
    Matrix44F space;
	space.setRotation(m_ccd.orientationB);
	space.setTranslation(m_ccd.positionB);
    
    Vector3F wb = space.transform(m_ccd.contactPointB);
	
	if(m_ccd.TOI == 0.f) {
		glColor3f(1.f, 0.f, 0.f);
		drawer->circleAt(wb, m_ccd.contactNormal);
	}
    
	glPushMatrix();
	drawer->useSpace(space);
	glColor3f(1.f, 0.f, 0.f);
	drawer->arrow(m_ccd.contactPointB, Vector3F::Zero);
	glBegin(GL_LINES);
	
	glColor3f(1.f, 0.f, 1.f);
	drawer->arrow(m_ccd.contactPointB, m_ccd.contactPointB + IinvJb);
	
	// glColor3f(0.f, 1.f, .3f);
	// glVertex3f(m_ccd.contactPointB.x, m_ccd.contactPointB.y, m_ccd.contactPointB.z);
	// glVertex3f(m_ccd.contactPointB.x + angularJ.x, m_ccd.contactPointB.y + angularJ.y, m_ccd.contactPointB.z + angularJ.z);
	glEnd();
	glPopMatrix();
#endif
}
