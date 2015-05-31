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
	
protected:
	virtual void rebuild(CudaLinearBvh * bvh);
    void * clusterAabbs();
private:
    int getM(int n, int m);
    unsigned sortPrimitives(CudaLinearBvh * bvh, 
                        unsigned numPrimitives, int n, int m);

	int splitClusters(CudaLinearBvh * bvh,
	                    unsigned numClusters);
	
	void decompressCluster(CudaLinearBvh * bvh, int numClusters, int numNodes);
	int splitPrimitives(CudaLinearBvh * bvh, int numInternal);

private:
    CUDABuffer * m_clusterAabb;
	CUDABuffer * m_runHeads;
	CUDABuffer * m_runIndices;
	CUDABuffer * m_runOffsets;
	CUDABuffer * m_runHash;
	CUDABuffer * m_runLength;
    CUDABuffer * m_queueAndElement;
};