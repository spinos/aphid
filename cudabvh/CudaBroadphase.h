/*
 *  CudaBroadphase.h
 *  cudabvh
 *
 *  Created by jian zhang on 2/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#define CUDABROADPHASE_MAX_NUMOBJECTS 32
class CUDABuffer;
class CudaLinearBvh;
class CudaBroadphase {
public:
	CudaBroadphase();
	virtual ~CudaBroadphase();
	
	void addBvh(CudaLinearBvh * bvh);
	void initOnDevice();
	void update();
protected:

private:
	void resetPairCounts();
	void countOverlappingPairs(unsigned a, unsigned b);
private:
	CUDABuffer * m_pairCounts;
	CudaLinearBvh * m_objects[CUDABROADPHASE_MAX_NUMOBJECTS];
	unsigned m_objectStart[CUDABROADPHASE_MAX_NUMOBJECTS];
	unsigned m_numObjects, m_numBoxes;
};