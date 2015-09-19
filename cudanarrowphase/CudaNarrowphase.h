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
class Vector3F;
class CudaMassSystem;
class CUDABuffer;
class BaseBuffer;
class CudaScan;
class CudaNarrowphase {
public:
	struct CombinedObjectBuffer {
		CUDABuffer * m_pos;
		CUDABuffer * m_pos0;
		CUDABuffer * m_prePos;
		CUDABuffer * m_vel;
        CUDABuffer * m_anchoredVel;
		CUDABuffer * m_mass;
		CUDABuffer * m_anchor;
        CUDABuffer * m_linearImpulse;
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
	void resetToInitial();
	void setAnchoredVelocity(Vector3F * src);
	
	void addMassSystem(CudaMassSystem * tetra);
	void computeContacts(CUDABuffer * overlappingPairBuf, unsigned numOverlappingPairs);
	
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

    void updateGravity(float dt);
	void upatePosition(float dt);
protected:

private:
	void resetContacts(void * overlappingPairs, unsigned numOverlappingPairs);
	void computeInitialSeparation();
	unsigned countNoPenetratingContacts(unsigned n);
	unsigned countPenetratingContacts(unsigned n);
	void computeTimeOfImpact();
	void handleShallowPenetrations();
	void squeezeContacts(unsigned numPairs);
private:
	CombinedObjectBuffer m_objectBuf;
	CUDABuffer * m_contact[2];
	CUDABuffer * m_contactPairs[2];
	CUDABuffer * m_validCounts;
	CUDABuffer * m_scanValidContacts;
	CudaScan * m_scanIntermediate;
	CUDABuffer * m_tetVertPos[2];
	CUDABuffer * m_tetVertPrePos;
	CUDABuffer * m_tetVertVel[2];
    CudaMassSystem * m_objects[CUDANARROWPHASE_MAX_NUMOBJECTS];
    unsigned m_objectPointStart[CUDANARROWPHASE_MAX_NUMOBJECTS];
	unsigned m_objectIndexStart[CUDANARROWPHASE_MAX_NUMOBJECTS];
	unsigned m_numObjects, m_numPoints, m_numElements, m_numContacts, m_numPairs;
};
#endif        //  #ifndef CUDANARROWPHASE_H

