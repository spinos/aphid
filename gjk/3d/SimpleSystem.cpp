/*
 *  SimpleSystem.cpp
 *  proof
 *
 *  Created by jian zhang on 1/25/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "SimpleSystem.h"
#define IDIM  10
#define IDIM1 11
// #define timeStep 0.0166667f
#define timeStep 0.0041667f
SimpleSystem::SimpleSystem()
{
	m_groundX = new Vector3F[IDIM1 * IDIM1];
	m_groundIndices = new unsigned[numGroundFaceVertices()];
	
	unsigned *ind = &m_groundIndices[0];
	unsigned i, j, i1, j1;
	for(j=0; j < IDIM; j++) {
	    j1 = j + 1;
		for(i=0; i < IDIM; i++) {
		    i1 = i + 1;
			*ind = j * IDIM1 + i;
			ind++;
			*ind = j1 * IDIM1 + i;
			ind++;
			*ind = j * IDIM1 + i1;
			ind++;

			*ind = j * IDIM1 + i1;
			ind++;
			*ind = j1 * IDIM1 + i;
			ind++;
			*ind = j1 * IDIM1 + i1;
			ind++;
		}
	}
	
	Vector3F * v = &m_groundX[0];
	for(j=0; j < IDIM1; j++) {
	    for(i=0; i < IDIM1; i++) {
		    i1 = i + 1;
			v->set(i * 3.f, -9.f, j * 3.f);
			v++;
		}
	}
	
	m_X = new Vector3F[3];
	m_X[0].set(1.1f, 1.3f, 10.8f);
	m_X[1].set(2.6f, 1.7f, 10.1f);
	m_X[2].set(2.5f, 4.3f, 10.4f);
	
	m_indices = new unsigned[3];
	m_indices[0] = 0;
	m_indices[1] = 1;
	m_indices[2] = 2;
	
	m_V = new Vector3F[3];
	m_Vline = new Vector3F[3 * 2];
	m_vIndices = new unsigned[3 * 2];
	
	for(i = 0; i< 3; i++) {
		m_V[i].set(10.f, 0.f, 0.f);
		m_Vline[i*2] = m_X[i];
		m_Vline[i*2 + 1] = m_X[i] + m_V[i] * timeStep;
		m_vIndices[i*2] = i*2;
		m_vIndices[i*2+1] = i*2+1;
	}
	
	m_rb.position.set(3.f, 16.f, 10.f);
	m_rb.orientation.set(1.f, 0.f, 0.f, 0.f);
	m_rb.linearVelocity.set(3.f, 0.f, 0.f);
	m_rb.angularVelocity.setZero();
	m_rb.shape = new CuboidShape(2.3f, 1.f, 2.f);
	
	m_ground.position.set(0.f, -10.f, 10.f);
	m_ground.orientation.set(1.f, 0.f, 0.f, 0.f);
	m_ground.linearVelocity.setZero();
	m_ground.angularVelocity.setZero();
	TetrahedronShape * tet = new TetrahedronShape;
	tet->p[0].set(0.f, 12.f, -20.f);
	tet->p[1].set(0.f, 12.f, 20.f);
	tet->p[2].set(40.f, 0.f, 0.f);
	tet->p[3].set(0.f, 0.f, -20.f);
	m_ground.shape = tet;
}

Vector3F * SimpleSystem::groundX() const
{ return m_groundX; }

const unsigned SimpleSystem::numGroundFaceVertices() const
{ return IDIM * IDIM * 2 * 3; }

unsigned * SimpleSystem::groundIndices() const
{ return m_groundIndices; }

Vector3F * SimpleSystem::X() const
{ return m_X; }

const unsigned SimpleSystem::numFaceVertices() const
{ return 3; }

unsigned * SimpleSystem::indices() const
{ return m_indices; }

Vector3F * SimpleSystem::Vline() const
{ return m_Vline; }

const unsigned SimpleSystem::numVlineVertices() const
{ return 3 * 2; }

unsigned * SimpleSystem::vlineIndices() const
{ return m_vIndices; }

void SimpleSystem::progress()
{
	int i;
	for(i = 0; i< 3; i++) {
		m_V[i] += Vector3F(0.f, -9.8f, 0.f) * timeStep;
	}
	
	for(i = 0; i< 3; i++) {
		m_X[i] += m_V[i] * timeStep;
	}
	
	for(i = 0; i< 3; i++) {
		m_Vline[i*2] = m_X[i];
		m_Vline[i*2 + 1] = m_X[i] + m_V[i] * timeStep;
	}
	
	applyGravity();
	applyImpulse();
	applyVelocity();
}

RigidBody * SimpleSystem::rb()
{ return &m_rb; }

RigidBody * SimpleSystem::ground()
{ return &m_ground; }

void SimpleSystem::applyGravity()
{ m_rb.linearVelocity += Vector3F(0.f, -9.8f, 0.f) * timeStep; }

void SimpleSystem::applyImpulse()
{
	continuousCollisionDetection(m_ground, m_rb);
	if(m_ccd.TOI == 0.f) return;
	
	
}

void SimpleSystem::continuousCollisionDetection(const RigidBody & A, const RigidBody & B)
{
	ContinuousCollisionContext &io = m_ccd;
	io.positionA = A.position;
	io.positionB = B.position;
	io.orientationA = A.orientation;
	io.orientationB = B.orientation;
	io.linearVelocityA = A.linearVelocity * timeStep;
	io.linearVelocityB = B.linearVelocity * timeStep;
	io.angularVelocityA = A.angularVelocity * timeStep;
	io.angularVelocityB = B.angularVelocity * timeStep;
	m_gjk.timeOfImpact(*A.shape, *B.shape, &m_ccd);
}

void SimpleSystem::applyVelocity()
{
	m_rb.position = m_rb.position.progress(m_rb.linearVelocity, timeStep);
	m_rb.orientation = m_rb.orientation.progress(m_rb.angularVelocity, timeStep);
}

void SimpleSystem::setDrawer(KdTreeDrawer * d)
{
	m_gjk.m_dbgDrawer = d;
}
