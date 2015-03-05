/*
 *  CudaNarrowphase.h
 *  testnarrowpahse
 *
 *  Created by jian zhang on 3/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#define CUDANARROWPHASE_MAX_NUMOBJECTS 32
class CudaTetrahedronSystem;
class CUDABuffer;
class BaseBuffer;
class CudaNarrowphase {
public:
	CudaNarrowphase();
	virtual ~CudaNarrowphase();
	
	void initOnDevice();
	
	void addTetrahedronSystem(CudaTetrahedronSystem * tetra);
	void computeContacts(CUDABuffer * overlappingPairBuf, unsigned numOverlappingPairs);
	
	void getSeparateAxis(BaseBuffer * dst);
	void getLocalA(BaseBuffer * dst);
	void getLocalB(BaseBuffer * dst);
	const unsigned numContacts() const;
protected:

private:
    void writeObjectCache(CudaTetrahedronSystem * tetra, 
	        unsigned pointAt, unsigned indexAt);
	void computeSeparateAxis(CUDABuffer * overlappingPairBuf, unsigned numOverlappingPairs);
private:
    CUDABuffer * m_pos;
    CUDABuffer * m_vel;
    CUDABuffer * m_ind;
	CUDABuffer * m_pointCacheLoc;
	CUDABuffer * m_indexCacheLoc;
	CUDABuffer * m_separateAxis;
	CUDABuffer * m_localA;
	CUDABuffer * m_localB;
    CudaTetrahedronSystem * m_objects[CUDANARROWPHASE_MAX_NUMOBJECTS];
    unsigned m_objectPointStart[CUDANARROWPHASE_MAX_NUMOBJECTS];
	unsigned m_objectIndexStart[CUDANARROWPHASE_MAX_NUMOBJECTS];
	unsigned m_numObjects, m_numPoints, m_numElements, m_numContacts;
};
