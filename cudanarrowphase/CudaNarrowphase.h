#ifndef CUDANARROWPHASE_H
#define CUDANARROWPHASE_H

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
	struct CombinedObjectBuffer {
		CUDABuffer * m_pos;
		CUDABuffer * m_vel;
		CUDABuffer * m_mass;
		CUDABuffer * m_ind;
		CUDABuffer * m_pointCacheLoc;
		CUDABuffer * m_indexCacheLoc;
	};
public:
	CudaNarrowphase();
	virtual ~CudaNarrowphase();
	
	const unsigned numPoints() const;
	const unsigned numElements() const;
	
	void initOnDevice();
	
	void addTetrahedronSystem(CudaTetrahedronSystem * tetra);
	void computeContacts(CUDABuffer * overlappingPairBuf, unsigned numOverlappingPairs);
	
	void getCoord(BaseBuffer * dst);
	void getContact(BaseBuffer * dst);
	const unsigned numPairs() const;
	const unsigned numContacts() const;
	
	void * contacts();
	void * contactPairs();
	
	void getContact0(BaseBuffer * dst);
	void * contacts0();
	
	void getContactCounts(BaseBuffer * dst);
	void getContactPairs(BaseBuffer * dst);
	void getScanResult(BaseBuffer * dst);
	
	CombinedObjectBuffer * objectBuffer();
	CUDABuffer * contactPairsBuffer();
	CUDABuffer * contactBuffer();
protected:

private:
	void computeTimeOfImpact(void * overlappingPairs, unsigned numOverlappingPairs);
	void squeezeContacts(void * overlappingPairs, unsigned numOverlappingPairs);
private:
	CombinedObjectBuffer m_objectBuf;
	CUDABuffer * m_coord;
	CUDABuffer * m_contact[2];
	CUDABuffer * m_contactPairs;
	CUDABuffer * m_validCounts;
	CUDABuffer * m_scanValidContacts[2];
    CudaTetrahedronSystem * m_objects[CUDANARROWPHASE_MAX_NUMOBJECTS];
    unsigned m_objectPointStart[CUDANARROWPHASE_MAX_NUMOBJECTS];
	unsigned m_objectIndexStart[CUDANARROWPHASE_MAX_NUMOBJECTS];
	unsigned m_numObjects, m_numPoints, m_numElements, m_numContacts, m_numPairs;
};
#endif        //  #ifndef CUDANARROWPHASE_H

