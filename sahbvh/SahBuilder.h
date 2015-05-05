/*
 *  SahBuilder.h
 *  testsah
 *
 *  Created by jian zhang on 5/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <BvhBuilder.h>
class CUDABuffer;
class SahBuilder : public BvhBuilder {
public:
	SahBuilder();
	virtual ~SahBuilder();
	
	virtual void initOnDevice();
	
	virtual void build(CudaLinearBvh * bvh);
protected:
	
private:
	void sortPrimitives(void * morton, unsigned numPrimitives, unsigned m);
private:
	CUDABuffer * m_runHeads;
	CUDABuffer * m_runIndices;
	CUDABuffer * m_compressedRunHeads;
	CUDABuffer * m_runHash;
	CUDABuffer * m_runLength;
	CUDABuffer * m_emissions[2];
};