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
	void * splitBins();
    void * splitIds();
    void * clusterAabbs();
    void * emissionBlocks();
private:
    int countTreeBits(void * morton, unsigned numPrimitives);
	int getM(int n, int m);
    unsigned sortPrimitives(void * morton, void * primitiveAabbs, 
                        unsigned numPrimitives, int n, int m);
private:
    CUDABuffer * m_mortonBits;
    CUDABuffer * m_clusterAabb;
	CUDABuffer * m_runHeads;
	CUDABuffer * m_runIndices;
	CUDABuffer * m_compressedRunHeads;
	CUDABuffer * m_runHash;
	CUDABuffer * m_runLength;
	CUDABuffer * m_emissions[2];
    CUDABuffer * m_splitBins;
    CUDABuffer * m_splitIds;
    CUDABuffer * m_emissionBlocks;
    CUDABuffer * m_numBinningBlocks;
};