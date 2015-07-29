/*
 *  MassSystem.cpp
 *  testcudafem
 *
 *  Created by jian zhang on 6/10/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "MassSystem.h"
#include <BaseBuffer.h>
#include <tetrahedron_math.h>
Vector3F MassSystem::WindVec = Vector3F::Zero;
MassSystem::MassSystem() : 
m_numTetrahedrons(0), 
m_numPoints(0), 
m_numTriangles(0),
m_totalMass(0.f)
{
	m_hostX = new BaseBuffer;
	m_hostXi = new BaseBuffer;
	m_hostV = new BaseBuffer;
	m_hostMass = new BaseBuffer;
	m_hostAnchor = new BaseBuffer;
	m_hostTetrahedronIndices = new BaseBuffer;
	m_hostTriangleIndices = new BaseBuffer;
}

MassSystem::~MassSystem() 
{
	delete m_hostX;
	delete m_hostXi;
	delete m_hostV;
    delete m_hostMass;
	delete m_hostAnchor;
	delete m_hostTetrahedronIndices;
	delete m_hostTriangleIndices;
}

void MassSystem::create(unsigned maxNumTetrahedrons, unsigned maxNumTriangles, unsigned maxNumPoints)
{
	m_maxNumPoints = maxNumPoints;
	m_maxNumTetrahedrons = maxNumTetrahedrons;
	m_maxNumTriangles = maxNumTriangles;
	m_hostX->create(m_maxNumPoints * 12);
	m_hostXi->create(m_maxNumPoints * 12);
	m_hostV->create(m_maxNumPoints * 12);
	m_hostMass->create(m_maxNumPoints * 4);
	m_hostAnchor->create(maxNumPoints * 4);
	m_hostTetrahedronIndices->create(m_maxNumTetrahedrons * 16);
	m_hostTriangleIndices->create(m_maxNumTriangles * 12);
}

void MassSystem::setNumPoints(unsigned x)
{ m_numPoints = x; }

void MassSystem::setNumTetrahedrons(unsigned x)
{ m_numTetrahedrons = x; }

void MassSystem::setNumTriangles(unsigned x)
{ m_numTriangles = x; }

const unsigned MassSystem::numPoints() const
{ return m_numPoints; }

const unsigned MassSystem::numTetrahedrons() const
{ return m_numTetrahedrons; }

const unsigned MassSystem::numTriangles() const
{ return m_numTriangles; }

const unsigned MassSystem::numTriangleFaceVertices() const
{ return m_numTriangles * 3; }

float * MassSystem::hostX()
{ return (float *)m_hostX->data(); }

float * MassSystem::hostXi()
{ return (float *)m_hostXi->data(); }

float * MassSystem::hostV()
{ return (float *)m_hostV->data(); }

float * MassSystem::hostMass()
{ return (float *)m_hostMass->data(); }

unsigned * MassSystem::hostAnchor()
{ return (unsigned *)m_hostAnchor->data(); }

unsigned * MassSystem::hostTriangleIndices()
{ return (unsigned *)m_hostTriangleIndices->data(); }

unsigned * MassSystem::hostTetrahedronIndices()
{ return (unsigned *)m_hostTetrahedronIndices->data(); }

const int MassSystem::elementRank() const
{ return 0; }

const unsigned MassSystem::numElements() const
{ return 0; }

void MassSystem::addPoint(float * src)
{
	if(m_numPoints == m_maxNumPoints) return;
	float * p = &hostX()[m_numPoints * 3];
	float * p0 = &hostXi()[m_numPoints * 3];
	unsigned * anchor = &hostAnchor()[m_numPoints];
	p[0] = src[0];
	p[1] = src[1];
	p[2] = src[2];
	p0[0] = p[0];
	p0[1] = p[1];
	p0[2] = p[2];
	*anchor = 0;
	m_numPoints++;
}

void MassSystem::addTetrahedron(unsigned a, unsigned b, unsigned c, unsigned d)
{
	if(m_numTetrahedrons == m_maxNumTetrahedrons) return;
	unsigned *idx = &hostTetrahedronIndices()[m_numTetrahedrons * 4];
	idx[0] = a;
	idx[1] = b;
	idx[2] = c;
	idx[3] = d;
	m_numTetrahedrons++;
	
	addTriangle(idx[TetrahedronToTriangleVertexByFace[0][0]], 
	            idx[TetrahedronToTriangleVertexByFace[0][1]],
	            idx[TetrahedronToTriangleVertexByFace[0][2]]);
	
	addTriangle(idx[TetrahedronToTriangleVertexByFace[1][0]], 
	            idx[TetrahedronToTriangleVertexByFace[1][1]],
	            idx[TetrahedronToTriangleVertexByFace[1][2]]);
	
	addTriangle(idx[TetrahedronToTriangleVertexByFace[2][0]], 
	            idx[TetrahedronToTriangleVertexByFace[2][1]],
	            idx[TetrahedronToTriangleVertexByFace[2][2]]);
	
	addTriangle(idx[TetrahedronToTriangleVertexByFace[3][0]], 
	            idx[TetrahedronToTriangleVertexByFace[3][1]],
	            idx[TetrahedronToTriangleVertexByFace[3][2]]);
}

void MassSystem::addTriangle(unsigned a, unsigned b, unsigned c)
{
	if(m_numTriangles == m_maxNumTriangles) return;
	unsigned *idx = &hostTriangleIndices()[m_numTriangles * 3];
	idx[0] = a;
	idx[1] = b;
	idx[2] = c;
	m_numTriangles++;
}

void MassSystem::resetVelocity()
{
	Vector3F * hv = (Vector3F *)hostV();
	const unsigned n = numPoints();
	unsigned i = 0;
	for(; i< n; i++) hv[i].setZero();
}

void MassSystem::setTotalMass(float x)
{ m_totalMass = x; }

const float MassSystem::totalMass() const
{ return m_totalMass; }

void MassSystem::setAnchoredValue(unsigned i, unsigned anchorInd)
{ hostAnchor()[i] = ((1<<24) | anchorInd); }

bool MassSystem::isAnchoredPoint(unsigned i)
{ return (hostAnchor()[i] > (1<<23)); }

void MassSystem::integrate(float dt) {}

void MassSystem::setHostX(float * src)
{ m_hostX->copyFrom(src, numPoints() * 12); }
//:~