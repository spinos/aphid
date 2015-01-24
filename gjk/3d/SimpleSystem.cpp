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
#define timeStep 0.0166667f
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
}
