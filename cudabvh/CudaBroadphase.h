#ifndef CUDABROADPHASE_H
#define CUDABROADPHASE_H

/*
 *  CudaBroadphase.h
 *  cudabvh
 *
 *  Created by jian zhang on 2/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <DynGlobal.h>

class BaseBuffer;
class CUDABuffer;
class CudaLinearBvh;
class CudaScan;
class CudaBroadphase {
public:
	CudaBroadphase();
	virtual ~CudaBroadphase();
	
	const unsigned numBoxes() const;
	const unsigned pairCacheLength() const;
	const unsigned objectStart(unsigned ind) const;
	const unsigned numOverlappingPairs() const;
	
	void addBvh(CudaLinearBvh * bvh);
	void initOnDevice();
	void computeOverlappingPairs();
	
	CUDABuffer * overlappingPairBuf();
	
	const unsigned numObjects() const;
	CudaLinearBvh * object(unsigned i) const;
	
	void sendDbgToHost();
	
#if DRAW_BPH_PAIRS
	void * hostPairCache();
	void * hostAabb();
#endif
protected:

private:
	void resetPairCounts();
	void countOverlappingPairs(unsigned a, unsigned b);
	void countOverlappingPairsSelf(unsigned a);
	void countOverlappingPairsOther(unsigned a, unsigned b);
	void setWriteLocation();
	void writeOverlappingPairs(unsigned a, unsigned b);
	void writeOverlappingPairsSelf(unsigned a);
	void writeOverlappingPairsOther(unsigned a, unsigned b);
	bool checkSystemRank(unsigned a);
private:
	CUDABuffer * m_pairCounts;
	CUDABuffer * m_pairStart;
	CUDABuffer * m_pairWriteLocation;
	CudaScan * m_scanIntermediate;
	CUDABuffer * m_pairCache;
#if DRAW_BPH_PAIRS
	BaseBuffer * m_hostPairCache;
	BaseBuffer * m_hostAabb;
#endif
	CudaLinearBvh * m_objects[CUDABROADPHASE_MAX_NUMOBJECTS];
	unsigned m_objectStart[CUDABROADPHASE_MAX_NUMOBJECTS];
	unsigned m_numObjects, m_numBoxes, m_scanBufferLength, m_pairCacheLength;
};
#endif        //  #ifndef CUDABROADPHASE_H
