/*
 *  BvhBuilder.h
 *  testsah
 *
 *  Created by jian zhang on 5/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
class CudaReduction;
class CudaLinearBvh;
class CUDABuffer;
class CudaScan;

class BvhBuilder {
public:
	BvhBuilder();
	virtual ~BvhBuilder();
	
	virtual void initOnDevice();
	
	virtual void build(CudaLinearBvh * bvh);
protected:
	CudaReduction * reducer();
	CudaScan * scanner();
	void createSortAndScanBuf(unsigned n);
	void computeMortionHash(void * mortonCode,
							void * primitiveAabbs, 
							unsigned numPrimitives,
							float * bounding);
	void sort(void * odata, unsigned nelem, unsigned nbits);
	void * sortIntermediate();
private:
	
private:
	CudaReduction * m_findMaxDistance;
	CudaScan * m_findPrefixSum;
	CUDABuffer * m_sortIntermediate;
};