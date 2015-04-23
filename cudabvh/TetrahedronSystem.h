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
#include <map>
class BaseBuffer;
struct TetrahedronMeshData;
class TetrahedronSystem {
public:
	TetrahedronSystem();
	virtual ~TetrahedronSystem();
	void generateFromData(TetrahedronMeshData * md);
	void create(const unsigned & maxNumTetrahedrons, const unsigned & maxNumPoints);
	void addPoint(float * src);
	void addTetrahedron(unsigned a, unsigned b, unsigned c, unsigned d);
	void setTotalMass(float x);
	void setAnchoredPoint(unsigned i, unsigned anchorInd);
	const unsigned numTetrahedrons() const;
	const unsigned numPoints() const;
	const unsigned numTriangles() const;
	const unsigned numTriangleFaceVertices() const;
	float * hostX();
	float * hostXi();
	float * hostV();
	float * hostMass();
	unsigned * hostAnchor();
	unsigned * hostTretradhedronIndices();
	unsigned * hostTriangleIndices();
	unsigned * hostTetrahedronVicinityInd();
	unsigned * hostTetrahedronVicinityStart();
protected:
	float totalInitialVolume();
    void calculateMass();
    const unsigned maxNumPoints() const;
	const unsigned maxNumTetradedrons() const;
	bool isAnchoredPoint(unsigned i);
	void createL1Vicinity();
	void createL2Vicinity();
	const unsigned numTetrahedronVicinityInd() const;
private:
    void addTriangle(unsigned a, unsigned b, unsigned c);
typedef std::map<unsigned, unsigned> VicinityMap;
typedef std::map<unsigned, unsigned>::iterator VicinityMapIter;
	void getPointTetrahedronConnection(VicinityMap * vertTetConn);
	void getTehrahedronTehrahedronConnectionL1(VicinityMap * tetTetConn, 
											VicinityMap * vertTetConn);
	void getTehrahedronTehrahedronConnectionL2(VicinityMap * dstConn, 
											VicinityMap * srcConn);
	void buildVicinityIndStart(VicinityMap * tetTetConn);
private:
	BaseBuffer * m_hostX;
	BaseBuffer * m_hostXi;
	BaseBuffer * m_hostV;
	BaseBuffer * m_hostMass;
	BaseBuffer * m_hostTetrahedronIndices;
	BaseBuffer * m_hostTriangleIndices;
	BaseBuffer * m_hostAnchor;
	BaseBuffer * m_hostTetrahedronVicinityInd;
	BaseBuffer * m_hostTetrahedronVicinityStart;
	unsigned m_numTetrahedrons, m_numPoints, m_numTriangles, m_tetrahedronVicinitySize;
	unsigned m_maxNumTetrahedrons, m_maxNumPoints, m_maxNumTriangles;
	float m_totalMass;
};
#endif        //  #ifndef TETRAHEDRONSYSTEM_H
