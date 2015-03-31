/*
 *  CudaBroadphase.h
 *  cudabvh
 *
 *  Created by jian zhang on 2/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#define CUDABROADPHASE_MAX_NUMOBJECTS 32
class BaseBuffer;
class CUDABuffer;
class CudaLinearBvh;
class CudaBroadphase {
public:
	CudaBroadphase();
	virtual ~CudaBroadphase();
	
	const unsigned numBoxes() const;
	const unsigned pairCacheLength() const;
	const unsigned objectStart(unsigned ind) const;
	const unsigned numUniquePairs() const;
	void getOverlappingPairCounts(BaseBuffer * dst);
	void getOverlappingPairCache(BaseBuffer * dst);
	void getScanCounts(BaseBuffer * dst);
	void getBoxes(BaseBuffer * dst);
	void getUniquePairs(BaseBuffer * dst);
	void getScanUniquePairs(BaseBuffer * dst);

	void addBvh(CudaLinearBvh * bvh);
	void initOnDevice();
	void computeOverlappingPairs();
	
	CUDABuffer * overlappingPairBuf();
protected:

private:
	void resetPairCounts();
	void countOverlappingPairs(unsigned a, unsigned b);
	void prefixSumPairCounts();
	unsigned getScanResult(CUDABuffer * counts, CUDABuffer * sums, unsigned n);
	void writeOverlappingPairs(unsigned a, unsigned b);
	void squeezeOverlappingPairs();
private:
	CUDABuffer * m_pairCounts;
	CUDABuffer * m_pairStart;
	CUDABuffer * m_scanIntermediate;
	CUDABuffer * m_pairCache[2];
	CUDABuffer * m_uniquePair;
	CUDABuffer * m_scanUniquePair;
	CudaLinearBvh * m_objects[CUDABROADPHASE_MAX_NUMOBJECTS];
	unsigned m_objectStart[CUDABROADPHASE_MAX_NUMOBJECTS];
	unsigned m_numObjects, m_numBoxes, m_scanBufferLength, m_pairCacheLength,
	        m_numUniquePairs;
};