/*
 *  TetrahedronSystem.cpp
 *  cudabvh
 *
 *  Created by jian zhang on 2/14/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "TetrahedronSystem.h"
#include <BaseBuffer.h>
TetrahedronSystem::TetrahedronSystem() :
m_numTetradedrons(0), m_numPoints(0), m_numTriangles(0)
{
	m_hostX = new BaseBuffer;
	m_hostV = new BaseBuffer;
	m_hostTretradhedronIndices = new BaseBuffer;
	m_hostTriangleIndices = new BaseBuffer;
}

TetrahedronSystem::~TetrahedronSystem() 
{
	delete m_hostX;
	delete m_hostV;
	delete m_hostTretradhedronIndices;
	delete m_hostTriangleIndices;
}

void TetrahedronSystem::create(unsigned maxNumTetrahedrons, float pointTetrahedronRatio, float triangleTetrahedronRatio)
{
	m_maxNumTetrahedrons = maxNumTetrahedrons;
	m_maxNumPoints = maxNumTetrahedrons * 4 * pointTetrahedronRatio;
	m_maxNumTriangles = maxNumTetrahedrons * 4 * triangleTetrahedronRatio;
	m_hostX->create(m_maxNumPoints * 12);
	m_hostV->create(m_maxNumPoints * 12);
	m_hostTretradhedronIndices->create(m_maxNumTetrahedrons * 16);
	m_hostTriangleIndices->create(m_maxNumTriangles * 12);
}

void TetrahedronSystem::addPoint(float * src)
{
	if(m_numPoints == m_maxNumPoints) return;
	float * p = &hostX()[m_numPoints * 3];
	p[0] = src[0];
	p[1] = src[1];
	p[2] = src[2];
	m_numPoints++;
}

void TetrahedronSystem::addTetrahedron(unsigned a, unsigned b, unsigned c, unsigned d)
{
	if(m_numTetradedrons == m_maxNumTetrahedrons) return;
	unsigned *idx = &hostTretradhedronIndices()[m_numTetradedrons * 4];
	idx[0] = a;
	idx[1] = b;
	idx[2] = c;
	idx[3] = d;
	m_numTetradedrons++;
}

void TetrahedronSystem::addTriangle(unsigned a, unsigned b, unsigned c)
{
	if(m_numTriangles == m_maxNumTriangles) return;
	unsigned *idx = &hostTriangleIndices()[m_numTriangles * 3];
	idx[0] = a;
	idx[1] = b;
	idx[2] = c;
	m_numTriangles++;
}

const unsigned TetrahedronSystem::numTetradedrons() const
{ return m_numTetradedrons; }

const unsigned TetrahedronSystem::numPoints() const
{ return m_numPoints; }

const unsigned TetrahedronSystem::numTriangles() const
{ return m_numTriangles; }

const unsigned TetrahedronSystem::maxNumPoints() const
{ return m_maxNumPoints; }

const unsigned TetrahedronSystem::maxNumTetradedrons() const
{ return m_maxNumTetrahedrons; }

const unsigned TetrahedronSystem::numTriangleFaceVertices() const
{ return m_numTriangles * 3; }

float * TetrahedronSystem::hostX()
{ return (float *)m_hostX->data(); }

float * TetrahedronSystem::hostV()
{ return (float *)m_hostV->data(); }

unsigned * TetrahedronSystem::hostTretradhedronIndices()
{ return (unsigned *)m_hostTretradhedronIndices->data(); }

unsigned * TetrahedronSystem::hostTriangleIndices()
{ return (unsigned *)m_hostTriangleIndices->data(); }
