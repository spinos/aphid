#ifndef TRIANGLESYSTEM_H
#define TRIANGLESYSTEM_H

#include "ATriangleMesh.h"
class BaseBuffer;
class TriangleSystem {
public:
    TriangleSystem(ATriangleMesh * md);
    virtual ~TriangleSystem();
    
    float * hostX();
	float * hostXi();
	float * hostV();
	float * hostMass();
	unsigned * hostTriangleIndices();
    unsigned * hostTetrahedronIndices();
    
    const unsigned numTriangles() const;
    const unsigned numPoints() const;
    const unsigned numTriangleFaceVertices() const;
protected:
    void create(unsigned numTri, unsigned numPnt);
private:
    BaseBuffer * m_hostX;
	BaseBuffer * m_hostXi;
	BaseBuffer * m_hostV;
	BaseBuffer * m_hostMass;
    BaseBuffer * m_hostTetrahedronIndices;
	BaseBuffer * m_hostTriangleIndices;
    unsigned m_numPoints, m_numTriangles;
};
#endif        //  #ifndef TRIANGLESYSTEM_H

