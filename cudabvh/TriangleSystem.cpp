#include "TriangleSystem.h"
#include <BaseBuffer.h>
TriangleSystem::TriangleSystem(ATriangleMesh * md) 
{    
    const unsigned np = md->numPoints();
    
    create(md->numTriangles(), md->numTriangles(), np);
    
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
	
	setNumPoints(np);
	setNumTetrahedrons(md->numTriangles());
	setNumTriangles(md->numTriangles());
}

TriangleSystem::~TriangleSystem() 
{
}

const int TriangleSystem::elementRank() const
{ return 3; }

const unsigned TriangleSystem::numElements() const
{ return numTriangles(); }
//:~