#ifndef TETRAHEDRONSYSTEM_H
#define TETRAHEDRONSYSTEM_H

/*
 *  TetrahedronSystem.h
 *  cudabvh
 *
 *  Created by jian zhang on 2/14/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
class BaseBuffer;
class TetrahedronSystem {
public:
	TetrahedronSystem();
	virtual ~TetrahedronSystem();
	void create(unsigned maxNumTetrahedrons, float pointTetrahedronRatio, float triangleTetrahedronRatio);
	void addPoint(float * src);
	void addTetrahedron(unsigned a, unsigned b, unsigned c, unsigned d);
	void addTriangle(unsigned a, unsigned b, unsigned c);
	const unsigned numTetradedrons() const;
	const unsigned numPoints() const;
	const unsigned numTriangles() const;
	const unsigned numTriangleFaceVertices() const;
	float * hostX();
	unsigned * hostTretradhedronIndices();
	unsigned * hostTriangleIndices();
protected:

private:
	BaseBuffer * m_hostX;
	BaseBuffer * m_hostTretradhedronIndices;
	BaseBuffer * m_hostTriangleIndices;
	unsigned m_numTetradedrons, m_numPoints, m_numTriangles;
	unsigned m_maxNumTetrahedrons, m_maxNumPoints, m_maxNumTriangles;
};
#endif        //  #ifndef TETRAHEDRONSYSTEM_H
