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
#include <CudaMassSystem.h>
class BaseBuffer;
class ATetrahedronMesh;
class TetrahedronSystem : public CudaMassSystem {
public:
	TetrahedronSystem();
	TetrahedronSystem(ATetrahedronMesh * md);
	virtual ~TetrahedronSystem();

	unsigned * hostTetrahedronVicinityInd();
	unsigned * hostTetrahedronVicinityStart();
	
// override mass system
	virtual const int elementRank() const;
	virtual const unsigned numElements() const;
protected:
	float totalInitialVolume();
    void calculateMass();
	void createL1Vicinity();
	void createL2Vicinity();
	const unsigned numTetrahedronVicinityInd() const;
private:
    
typedef std::map<unsigned, unsigned> VicinityMap;
typedef std::map<unsigned, unsigned>::iterator VicinityMapIter;
	void getPointTetrahedronConnection(VicinityMap * vertTetConn);
	void getTehrahedronTehrahedronConnectionL1(VicinityMap * tetTetConn, 
											VicinityMap * vertTetConn);
	void getTehrahedronTehrahedronConnectionL2(VicinityMap * dstConn, 
											VicinityMap * srcConn);
	void buildVicinityIndStart(VicinityMap * tetTetConn);
private:
	BaseBuffer * m_hostTetrahedronVicinityInd;
	BaseBuffer * m_hostTetrahedronVicinityStart;
	unsigned m_tetrahedronVicinitySize;
};
#endif        //  #ifndef TETRAHEDRONSYSTEM_H
