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
#include <tetrahedron_math.h>
TetrahedronSystem::TetrahedronSystem() :
m_numTetrahedrons(0), m_numPoints(0), m_numTriangles(0)
{
	m_hostX = new BaseBuffer;
	m_hostXi = new BaseBuffer;
	m_hostV = new BaseBuffer;
	m_hostTretradhedronIndices = new BaseBuffer;
	m_hostTriangleIndices = new BaseBuffer;
	m_hostMass = new BaseBuffer;
	m_hostAnchor = new BaseBuffer;
	m_totalMass = 10.f;
}

TetrahedronSystem::~TetrahedronSystem() 
{
	delete m_hostX;
	delete m_hostXi;
	delete m_hostV;
	delete m_hostTretradhedronIndices;
	delete m_hostTriangleIndices;
	delete m_hostAnchor;
}

void TetrahedronSystem::create(const unsigned & maxNumTetrahedrons, const unsigned & maxNumPoints)
{
	m_maxNumTetrahedrons = maxNumTetrahedrons;
	m_maxNumPoints = maxNumPoints;
	m_maxNumTriangles = maxNumTetrahedrons * 4;
	m_hostX->create(m_maxNumPoints * 12);
	m_hostXi->create(m_maxNumPoints * 12);
	m_hostV->create(m_maxNumPoints * 12);
	m_hostMass->create(m_maxNumPoints * 4);
	m_hostAnchor->create(m_maxNumPoints * 4);
	m_hostTretradhedronIndices->create(m_maxNumTetrahedrons * 16);
	m_hostTriangleIndices->create(m_maxNumTriangles * 12);
}

void TetrahedronSystem::setTotalMass(float x)
{ m_totalMass = x; }

void TetrahedronSystem::addPoint(float * src)
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

void TetrahedronSystem::addTetrahedron(unsigned a, unsigned b, unsigned c, unsigned d)
{
	if(m_numTetrahedrons == m_maxNumTetrahedrons) return;
	unsigned *idx = &hostTretradhedronIndices()[m_numTetrahedrons * 4];
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

void TetrahedronSystem::addTriangle(unsigned a, unsigned b, unsigned c)
{
	if(m_numTriangles == m_maxNumTriangles) return;
	unsigned *idx = &hostTriangleIndices()[m_numTriangles * 3];
	idx[0] = a;
	idx[1] = b;
	idx[2] = c;
	m_numTriangles++;
}

const unsigned TetrahedronSystem::numTetrahedrons() const
{ return m_numTetrahedrons; }

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

float * TetrahedronSystem::hostXi()
{ return (float *)m_hostXi->data(); }

float * TetrahedronSystem::hostV()
{ return (float *)m_hostV->data(); }

float * TetrahedronSystem::hostMass()
{ return (float *)m_hostMass->data(); }

unsigned * TetrahedronSystem::hostAnchor()
{ return (unsigned *)m_hostAnchor->data(); }

unsigned * TetrahedronSystem::hostTretradhedronIndices()
{ return (unsigned *)m_hostTretradhedronIndices->data(); }

unsigned * TetrahedronSystem::hostTriangleIndices()
{ return (unsigned *)m_hostTriangleIndices->data(); }

float TetrahedronSystem::totalInitialVolume()
{
	Vector3F * p = (Vector3F *)hostXi();
    unsigned * v = hostTretradhedronIndices();
    unsigned i;
	Vector3F t[4];
	unsigned a, b, c, d;
	float sum = 0.f;
	for(i=0; i<m_numTetrahedrons; i++) {
		a = v[0];
		b = v[1];
		c = v[2];
		d = v[3];
		t[0] = p[a];
		t[1] = p[b];
		t[2] = p[c];
		t[3] = p[d];
		sum += tetrahedronVolume(t);
	}
	return sum;
}

void TetrahedronSystem::calculateMass()
{
	float density = m_totalMass / totalInitialVolume();
	if(density < 0.f) density = -density;
	std::cout<<" density "<<density;
    const float base = 1.f/(float)m_numPoints;
    unsigned i;
    float * mass = hostMass();
    for(i=0; i< m_numPoints; i++) {
		if(isAnchoredPoint(i))
			mass[i] = 1e30f;
        else
			mass[i] = base;
    }
    
    Vector3F * p = (Vector3F *)hostXi();
    
    Vector3F v[4];
    unsigned a, b, c, d;
    unsigned *ind = hostTretradhedronIndices();
    float m;
    for(i=0; i<m_numTetrahedrons; i++) {
		a = ind[0];
		b = ind[1];
		c = ind[2];
		d = ind[3];
		
		v[0] = p[a];
		v[1] = p[b];
		v[2] = p[c];
		v[3] = p[d];
		
		m = density * tetrahedronVolume(v) * .25f;
		
		mass[a] += m;
		mass[b] += m;
		mass[c] += m;
		mass[d] += m;
		
		ind += 4;
	}
	/*
	for(i=0; i< m_numPoints; i++) {
	    std::cout<<" m "<<mass[i];
    }
    */
}

void TetrahedronSystem::setAnchoredPoint(unsigned i, unsigned anchorInd)
{
	unsigned * anchor = &hostAnchor()[i];
	*anchor = ((1<<30) | anchorInd);
}

bool TetrahedronSystem::isAnchoredPoint(unsigned i)
{ return (hostAnchor()[i] > (1<<29)); }
