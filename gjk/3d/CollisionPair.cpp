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
	m_ccd.positionB = m_B->position.progress(m_B->linearVelocity, h);
	m_ccd.orientationB = m_B->orientation.progress(m_B->angularVelocity, h);
}

void CollisionPair::progressOnImpactPostion(const float & h)
{
// only B at impact position
	m_ccd.positionB = m_B->position.progress(m_B->projectedLinearVelocity, h);
	m_ccd.orientationB = m_B->orientation.progress(m_B->projectedAngularVelocity, h);
}

void CollisionPair::detectAtImpactPosition(const float & h)
{
	m_ccd.linearVelocityB = m_B->projectedLinearVelocity * h;
	m_ccd.angularVelocityB = m_B->projectedAngularVelocity * h;
	m_gjk.timeOfImpact(*m_A->shape, *m_B->shape, &m_ccd);
	
	std::cout<<"\nb test v"<<m_ccd.linearVelocityB * 60.f;
	std::cout<<"\nb test w"<<m_ccd.angularVelocityB * 60.f;
	std::cout<<" toi"<<m_ccd.TOI;
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
	Vector3F Vrel = relativeLinearVelocity();//relativeVelocity();//
	
	const float massinvA = 0.f;
	const float massinvB = m_B->shape->linearMassM.x;
	
	float MinvJ = Vrel.dot(m_ccd.contactNormal) / (massinvA + massinvB);
	
	MinvJa = -(1.f + m_A->Crestitution) * MinvJ * massinvA;
	MinvJb = (1.f + m_B->Crestitution) * MinvJ * massinvB;
	if(m_ccd.penetrateDepth > 0.f) {
		// std::cout<<" penetrate d add relative velocity"<<m_ccd.penetrateDepth;
		MinvJb += m_ccd.penetrateDepth * 60.f;
	}
	// std::cout<<" MinvJb "<<MinvJb;
	N = m_ccd.contactNormal;
}

const Vector3F CollisionPair::relativeVelocity() const
{
	return velocityAtContactA() - velocityAtContactB();
}

const Vector3F CollisionPair::relativeLinearVelocity() const
{
	return Vector3F::Zero - m_B->projectedLinearVelocity;
}

const Vector3F CollisionPair::relativeAngularVelocity() const
{
	return Vector3F::Zero - angularMotionAtContactB();
}

const Vector3F CollisionPair::velocityAtContactA() const
{
	return Vector3F::Zero;
}

const Vector3F CollisionPair::velocityAtContactB() const
{
	return m_B->projectedLinearVelocity + angularMotionAtContactB();
}

const Vector3F CollisionPair::angularMotionAtContactB() const
{
	Matrix44F R;
	getTransformB(R);
	Vector3F ro = m_ccd.contactPointB;
#ifdef DBG_DRAW
	KdTreeDrawer * drawer = m_gjk.m_dbgDrawer;	
	glColor3f(.3, .5, .8);
	Vector3F wp = R.transform(m_ccd.contactPointB);
	drawer->arrow(wp, wp + R.transformAsNormal(ro.cross(m_B->projectedAngularVelocity)));
#endif
	return R.transformAsNormal(ro.cross(m_B->projectedAngularVelocity));
}

void CollisionPair::computeAngularImpulse(Vector3F & IinvJa, float & MinvJa, Vector3F & IinvJb, float & MinvJb)
{
	Vector3F Vrel = relativeAngularVelocity();
	if(Vrel.dot(m_ccd.contactNormal) < TINY_VALUE) {
		Vrel = relativeLinearVelocity();
	}
	
	const float maxTorqueSize = m_ccd.contactPointB.length() * 2.f;
	float vdotn = Vrel.dot(m_ccd.contactNormal);
	if(vdotn > maxTorqueSize) vdotn = maxTorqueSize;
	if(vdotn < -maxTorqueSize) vdotn = -maxTorqueSize;
	
	Matrix44F R; 
	getTransformB(R);
	Matrix44F Ri = R;
	Ri.inverse();
	
	std::cout<<" b at "<<m_ccd.positionB<<"\n";
	std::cout<<" real at"<<m_B->position<<"\n";
	std::cout<<" contact p"<<m_ccd.contactPointB.str()<<"\n";
	std::cout<<" linear v"<<m_B->projectedLinearVelocity.str()<<"\n";
	std::cout<<" angular w"<<m_B->projectedAngularVelocity.str()<<"\n";
	std::cout<<" angular motion"<<angularMotionAtContactB().str()<<"\n";
	std::cout<<" linear motion rel"<<m_B->projectedLinearVelocity.dot(m_ccd.contactNormal)<<"\n";
	std::cout<<" angular motion rel"<<angularMotionAtContactB().dot(m_ccd.contactNormal)<<"\n";
	std::cout<<" relative vel"<<Vrel.dot(m_ccd.contactNormal)<<"\n";
	
// N in object space
	Vector3F nb = Ri.transformAsNormal(m_ccd.contactNormal);

// from contact point to center in object space	
	const Vector3F rb = m_ccd.contactPointB.reversed();

// torque in object space	
	IinvJb = m_B->shape->angularMassM * rb.cross(nb);
	//rb.cross(nb).verbose(" angular impulse ");
	//IinvJb.verbose(" angular impulse / I");
	//std::cout<<"1/I"<<m_B->shape->angularMassM.str();
		
	const float massinv = m_B->shape->linearMassM.x;
	
	const Vector3F wr = R.transformAsNormal(m_ccd.contactPointB.reversed());
	
	
	const float MinvJ =  vdotn / (massinv + (m_B->inertiaTensor * wr.cross(m_ccd.contactNormal)).cross(wr).dot(m_ccd.contactNormal));

	MinvJb = (1.f + 1.f) * MinvJ;
	//std::cout<<" dot "<<Vrel.dot(m_ccd.contactNormal);
	
	//std::cout<<" size "<<MinvJb;
	//std::cout<<" tensor "<<(m_B->inertiaTensor * wr.cross(m_ccd.contactNormal)).str();
	
#ifdef DBG_DRAW
	m_B->r = m_ccd.contactPointB;
	m_B->J = IinvJb;
	m_B->Jsize = MinvJb;
    KdTreeDrawer * drawer = m_gjk.m_dbgDrawer;
	
    Matrix44F space; 
	getTransformB(space);
	
    Vector3F wp = space.transform(m_ccd.contactPointB);
	if(m_ccd.TOI == 0.f) {
		glColor3f(1.f, 0.f, 0.f);
		drawer->circleAt(wp, m_ccd.contactNormal);
	}
	
	glColor3f(0,1,1);
	drawer->arrow(wp, wp + m_ccd.contactNormal * Vrel);
	glColor3f(1,0,1);
	// drawer->arrow(m_ccd.contactPointB, m_ccd.contactPointB + Vrel);
	
	glPushMatrix();
	drawer->useSpace(space);
	glColor3f(1.f, 0.f, 0.f);
	drawer->arrow(m_ccd.contactPointB, Vector3F::Zero);
    
	glColor3f(.5f, 0.f, 0.f);
	drawer->arrow(m_ccd.contactPointB - rb.cross(nb) * .5f, m_ccd.contactPointB + rb.cross(nb) * .5f);
    
	// glBegin(GL_LINES);
	
	// glColor3f(1.f, 0.f, 1.f);
	// drawer->arrow(m_ccd.contactPointB, m_ccd.contactPointB + IinvJb);
	
	// glColor3f(0.f, 1.f, .3f);
	// glVertex3f(m_ccd.contactPointB.x, m_ccd.contactPointB.y, m_ccd.contactPointB.z);
	// glVertex3f(m_ccd.contactPointB.x + angularJ.x, m_ccd.contactPointB.y + angularJ.y, m_ccd.contactPointB.z + angularJ.z);
	// glEnd();
	glPopMatrix();
#endif
}

void CollisionPair::getTransformA(Matrix44F & t) const
{
	t.setRotation(m_ccd.orientationA);
	t.setTranslation(m_ccd.positionA);
}
	
void CollisionPair::getTransformB(Matrix44F & t) const
{
	t.setRotation(m_ccd.orientationB);
	t.setTranslation(m_ccd.positionB);
}
