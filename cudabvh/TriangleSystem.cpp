#include "TriangleSystem.h"
#include <BaseBuffer.h>
TriangleSystem::TriangleSystem(ATriangleMesh * md) 
{
   m_hostX = new BaseBuffer;
	m_hostXi = new BaseBuffer;
	m_hostV = new BaseBuffer;
	m_hostTriangleIndices = new BaseBuffer;
    m_hostTetrahedronIndices = new BaseBuffer;
	m_hostMass = new BaseBuffer;
    
    const unsigned np = md->numPoints();
    
    create(md->numTriangles(), np);
    
    Vector3F * p = md->points();
    Vector3F * q = (Vector3F *)hostX();
    unsigned i = 0;
    for(;i<np;i++) q[i] = p[i];
    
    Vector3F * qi = (Vector3F *)hostXi();
    for(i=0;i<np;i++) qi[i] = q[i];
    
    Vector3F * vel = (Vector3F *)hostV();
    for(i=0;i<np;i++) vel[i].setZero();
    
    unsigned * ind = (unsigned *)md->indices();
    unsigned * tris = hostTriangleIndices();
    for(i=0; i< md->numIndices(); i++)
        tris[i] = ind[i];
    
    unsigned * tetra = hostTetrahedronIndices();
    for(i=0; i< md->numTriangles(); i++) {
        tetra[i*4] = ind[i*3];
        tetra[i*4+1] = ind[i*3+1];
        tetra[i*4+2] = ind[i*3+2];
        tetra[i*4+3] = tetra[i*4+2];
    }
    
    float * mass = hostMass();
    for(i=0;i<np;i++) mass[i] = 1e30f;
}

TriangleSystem::~TriangleSystem() 
{
    delete m_hostX;
	delete m_hostXi;
	delete m_hostV;
    delete m_hostMass;
	delete m_hostTriangleIndices;
}

void TriangleSystem::create(unsigned numTri, unsigned numPnt)
{
    m_numTriangles = numTri;
    m_numPoints = numPnt;
    m_hostX->create(m_numPoints * 12);
	m_hostXi->create(m_numPoints * 12);
	m_hostV->create(m_numPoints * 12);
	m_hostMass->create(m_numPoints * 4);
	m_hostTriangleIndices->create(m_numTriangles * 12);
    m_hostTetrahedronIndices->create(m_numTriangles * 16);
}

const unsigned TriangleSystem::numTriangles() const
{ return m_numTriangles; }

const unsigned TriangleSystem::numPoints() const
{ return m_numPoints; }

const unsigned TriangleSystem::numTriangleFaceVertices() const
{ return m_numTriangles * 3; }

float * TriangleSystem::hostX()
{ return (float *)m_hostX->data(); }

float * TriangleSystem::hostXi()
{ return (float *)m_hostXi->data(); }

float * TriangleSystem::hostV()
{ return (float *)m_hostV->data(); }

float * TriangleSystem::hostMass()
{ return (float *)m_hostMass->data(); }

unsigned * TriangleSystem::hostTriangleIndices()
{ return (unsigned *)m_hostTriangleIndices->data(); }

unsigned * TriangleSystem::hostTetrahedronIndices()
{ return (unsigned *)m_hostTetrahedronIndices->data(); }
