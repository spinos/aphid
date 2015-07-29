/*
 *  SahBuilder.cpp
 *  testsah
 *
 *  Created by jian zhang on 5/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
input primitive range 0,9
node 0

node_count = 1 (root)
N_segments = 1

N=10 input primitives
0 1 2 3 4 5 6 7 8 9 

morton_code[N]
0 0 0 0 1 1 1 1 1 1
0 1 1 1 0 0 1 1 1 1
1 0 1 1 1 1 0 0 1 1

head_to_node[N]
0 -1 -1 -1 -1 -1 -1 -1 -1

segment_heads[N]
0 0 0 0 0 0 0 0 0 0

segment_id[N]
0 0 0 0 0 0 0 0 0 0 

N_segments = 1 + segment_id[N-1]

P = (1<<p) - 1 where p=3

block_splits[N * 3] = {-1, -1, -1, ...}

block_splits[N*3]
0 1 2 3 4 5 6 7 8 9
        1           level1
  1         1       level2
    1           1   level3

segment_heads[N]
0 1 1 0 1 0 1 0 1 0 combined


segment_id[N]
0 1 2 2 3 3 4 4 5 5 inclusive_scan(segment_heads)

N_segments = 1 + 5

head_to_node[N]
0 1 2 -1 4 -1 6 -1 8 -1

N_splits = 5 sum(segment_heads)

node_count += N_splits * 2

block_offset[N_segments + 1]
0 1 2 3 4 5 6 7 8 9
        1          
  1         1      
    1           1   
1 1 1   1   1   1
       0           
  1         2      
    3           4  
5 6 7  8    9   10

block_offset by scan block_splits and head_to_node>-1

internal node (child0, child1)

0: 1,2   
1: 5,3   
2: 8,4   
3: 6,7
4: 9,10

leaf node (primitive0, primitive1)
5: 0,0
6: 1,1
7: 2,3
8: 4,5
9: 6,7
10: 8,9

3 level, max 7 splits, max 6 new internal nodes, max 8 leaf nodes

n splits spawn n-1 internal nodes, n+1 leaf nodes, n+1 segments, 
each segment will be input primitive range of next 3 level splits

each node holds segment begin/end initially. 
internal nodes will look for child nodes based on correct segment begin/end. For example.

node1 should connect to node3 and node4 by default, 
but non-split leads to node3 begin with 1, so go to next node, 
until find node5 begin with 0. then looking for first node end with 3, 
which is node3. So node1 has child node5 and node3 

input n_blocks
	root_id
      block_offset (relative to last node)

output:
n_blocks 1
root_id 0
block_offset 0
spawn node  1 2 3 4 5 6 7 8 9 10

output:
n_blocks 6
root_id (leaf) 5 6 7 8 9 10
block_offset 11 25 39 53 67 81

give 14 nodes to each block
 *
 */

#include "SahBuilder.h"
#include <CudaDbgLog.h>
#include <CudaReduction.h>
#include <CudaLinearBvh.h>
#include <CudaBase.h>
#include <CudaScan.h>
#include "SahInterface.h"

// CudaDbgLog sahlg("sah.txt");

static std::vector<std::pair<int, int> > binDesc;
static std::vector<std::pair<int, int> > emissionDesc;
static std::vector<std::pair<int, int> > emissionBlockDesc;

SahBuilder::SahBuilder() 
{
    binDesc.push_back(std::pair<int, int>(1, 0));
	binDesc.push_back(std::pair<int, int>(1, 4));
	binDesc.push_back(std::pair<int, int>(1, 8));
	binDesc.push_back(std::pair<int, int>(1, 12));
	binDesc.push_back(std::pair<int, int>(1, 16));
	binDesc.push_back(std::pair<int, int>(1, 20));
	binDesc.push_back(std::pair<int, int>(0, 24));
	binDesc.push_back(std::pair<int, int>(1, 28));
	binDesc.push_back(std::pair<int, int>(1, 32));
	binDesc.push_back(std::pair<int, int>(1, 36));
	binDesc.push_back(std::pair<int, int>(1, 40));
	binDesc.push_back(std::pair<int, int>(1, 44));
	binDesc.push_back(std::pair<int, int>(1, 48));
	binDesc.push_back(std::pair<int, int>(0, 52));
	binDesc.push_back(std::pair<int, int>(0, 56));
	binDesc.push_back(std::pair<int, int>(1, 60));
	
	emissionDesc.push_back(std::pair<int, int>(0, 0));
	emissionDesc.push_back(std::pair<int, int>(0, 4));
	emissionDesc.push_back(std::pair<int, int>(0, 8));
    emissionDesc.push_back(std::pair<int, int>(0, 12));
    
    emissionBlockDesc.push_back(std::pair<int, int>(0, 0));
	emissionBlockDesc.push_back(std::pair<int, int>(0, 4));
	emissionBlockDesc.push_back(std::pair<int, int>(0, 8));
    emissionBlockDesc.push_back(std::pair<int, int>(0, 12));
	
    m_clusterAabb = new CUDABuffer;
	m_runHeads = new CUDABuffer;
	m_runIndices = new CUDABuffer;
	m_runHash = new CUDABuffer;
	m_runLength = new CUDABuffer;
	m_runOffsets = new CUDABuffer;
	m_queueAndElement = new CUDABuffer;
}

SahBuilder::~SahBuilder() 
{
    delete m_clusterAabb;
	delete m_runHeads;
	delete m_runIndices;
	delete m_runHash;
	delete m_runLength;
	delete m_runOffsets;
	delete m_queueAndElement;
}

void SahBuilder::initOnDevice()
{
    BvhBuilder::initOnDevice();
}

void SahBuilder::rebuild(CudaLinearBvh * bvh)
{
	const unsigned n = bvh->numPrimitives();
	createSortAndScanBuf(n);
	
    m_runHeads->create(CudaScan::getScanBufferLength(n) * 4);
	m_runIndices->create(CudaScan::getScanBufferLength(n) * 4);
	m_clusterAabb->create(n * 24);
	
	void * primitiveHash = bvh->primitiveHash();
	void * primitiveAabb = bvh->primitiveAabb();
	
	float bounding[6];
	computeMortionHash(primitiveHash, primitiveAabb, n, bounding);
	
	unsigned numClusters = 0;
	numClusters = sortPrimitives(bvh, n, 10, 5);
    
    int rr[2];
    rr[0] = 0;
    rr[1] = numClusters - 1;
	bvh->initRootNode(rr, bounding);
    
    m_queueAndElement->create(SIZE_OF_SIMPLEQUEUE + n * 4);
    int numInternal = splitClusters(bvh, numClusters);
    decompressCluster(bvh, numClusters, numInternal);
    numInternal = splitPrimitives(bvh, numInternal);
    bvh->setNumActiveInternalNodes(numInternal);
    
    // sahlg.writeInt(bvh->distanceInternalNodeFromRootBuf(), numInternal, "node_level", CudaDbgLog::FOnce);
    
    int maxDistanceToRoot = 0;
	reducer()->max<int>(maxDistanceToRoot, (int *)bvh->distanceInternalNodeFromRoot(), numInternal);
	std::cout<<" bvh max level "<<maxDistanceToRoot<<"\n";
	bvh->setMaxInternalNodeLevel(maxDistanceToRoot);
	
	countPrimitivesInNode(bvh);
    
    // sahlg.writeInt(bvh->internalNodeNumPrimitiveBuf(), numInternal, "primitive_count", CudaDbgLog::FOnce);
	
	float cost = computeCostOfTraverse(bvh);
    std::cout<<" cost of traverse: "<<cost<<"\n";
    bvh->setCostOfTraverse(cost);
}

int SahBuilder::splitClusters(CudaLinearBvh * bvh, unsigned numClusters)
{
    int n = sahsplit::doSplitWorks(m_queueAndElement->bufferOnDevice(),
        (int *)m_queueAndElement->bufferOnDeviceAt(SIZE_OF_SIMPLEQUEUE),
        (int2 *)bvh->internalNodeChildIndices(),
	    (Aabb *)bvh->internalNodeAabbs(),
        (int *)bvh->internalNodeParentIndices(),
        (int *)bvh->distanceInternalNodeFromRoot(),
	    (KeyValuePair *)m_runHash->bufferOnDevice(),
        (Aabb *)clusterAabbs(),
        (KeyValuePair *)m_runHash->bufferOnDeviceAt(numClusters * 8),
        numClusters,
        1);
    
    // sahlg.writeHash(m_runHash, numClusters, "split_indirections", CudaDbgLog::FOnce);
    // sahlg.writeInt2(bvh->internalChildBuf(), n, "internal_node", CudaDbgLog::FOnce);
    // sahlg.writeAabb(bvh->internalAabbBuf(), n, "internal_box", CudaDbgLog::FOnce);
    // sahlg.writeUInt(bvh->internalParentBuf(), n, "parent_node", CudaDbgLog::FOnce);
    return n;
}
 
void SahBuilder::decompressCluster(CudaLinearBvh * bvh, int numClusters, int numNodes)
{
    sahdecompress::countLeaves((uint *)m_runLength->bufferOnDevice(),
                                (int *)m_queueAndElement->bufferOnDeviceAt(SIZE_OF_SIMPLEQUEUE),
                                (int2 *)bvh->internalNodeChildIndices(),
                                (KeyValuePair *)m_runHash->bufferOnDevice(),
                                (uint *)m_runOffsets->bufferOnDevice(),
                                numClusters,
                                bvh->numPrimitives(),
                                numNodes,
                                CudaScan::getScanBufferLength(numNodes));
							
	// sahlg.writeUInt(m_runLength, numNodes, "leaf_length", CudaDbgLog::FOnce);

    scanner()->prefixSum(m_runIndices, m_runLength, CudaScan::getScanBufferLength(numNodes));
	
	// sahlg.writeUInt(m_queueAndElement, bvh->numPrimitives(), "queue_lock", CudaDbgLog::FOnce);

    sahdecompress::copyHash((KeyValuePair *)sortIntermediate(),
					(KeyValuePair *)bvh->primitiveHash(),
					bvh->numPrimitives());
	
	sahdecompress::decompressPrimitives((KeyValuePair *)bvh->primitiveHash(),
                            (KeyValuePair *)sortIntermediate(),
                            (int2 *)bvh->internalNodeChildIndices(),
                            (KeyValuePair *)m_runHash->bufferOnDevice(),
                            (uint *)m_runIndices->bufferOnDevice(),
                            (uint *)m_runOffsets->bufferOnDevice(),
                            numClusters,
                            bvh->numPrimitives(),
                            numNodes);
    // sahlg.writeInt2(bvh->internalChildBuf(), numNodes, "decompressed_node", CudaDbgLog::FOnce); 
}

int SahBuilder::splitPrimitives(CudaLinearBvh * bvh, int numInternal)
{
    int nn = sahsplit::doSplitWorks(m_queueAndElement->bufferOnDevice(),
        (int *)m_queueAndElement->bufferOnDeviceAt(SIZE_OF_SIMPLEQUEUE),
        (int2 *)bvh->internalNodeChildIndices(),
	    (Aabb *)bvh->internalNodeAabbs(),
        (int *)bvh->internalNodeParentIndices(),
        (int *)bvh->distanceInternalNodeFromRoot(),
	    (KeyValuePair *)bvh->primitiveHash(),
        (Aabb *)bvh->primitiveAabb(),
        (KeyValuePair *)sortIntermediate(),
        bvh->numPrimitives(),
        numInternal); 
    // sahlg.writeInt2(bvh->internalChildBuf(), nn, "internal_node1", CudaDbgLog::FOnce);
    // sahlg.writeUInt(bvh->internalParentBuf(), x, "parent_node1", CudaDbgLog::FOnce);
    return nn;
}

int SahBuilder::getM(int n, int m)
{
    int r = n - 3 * m;
    while(r<=0) {
        m--;
        r = n - 3 * m;
    }
    return m;
}

#define COMPRESS_TWICE 1

unsigned SahBuilder::sortPrimitives(CudaLinearBvh * bvh, 
                                unsigned numPrimitives, int n, int m)
{
    void * primitiveHash = bvh->primitiveHash();
	void * primitiveAabb = bvh->primitiveAabb();
	
    const unsigned scanLength = CudaScan::getScanBufferLength(numPrimitives);
	
	const unsigned d = 3*(n - m);

	sahcompress::computeRunHead((uint *)m_runHeads->bufferOnDevice(), 
							(KeyValuePair *)primitiveHash,
							d,
							numPrimitives,
							scanLength);
	
	// CudaBase::CheckCudaError("finding bvh run heads");
	
	// sahlg.writeUInt(m_runHeads, numPrimitives, "run_heads", CudaDbgLog::FOnce);

	unsigned numRuns = scanner()->prefixSum(m_runIndices, 
												m_runHeads, scanLength);
												
	m_runOffsets->create(numRuns * 4);
	
	sahcompress::compressRunHead((uint *)m_runOffsets->bufferOnDevice(), 
							(uint *)m_runHeads->bufferOnDevice(),
							(uint *)m_runIndices->bufferOnDevice(),
							numPrimitives);

    m_runHash->create((nextPow2(numRuns) * 2 * 8));
 
    sahcompress::computeRunHash((KeyValuePair *)m_runHash->bufferOnDevice(), 
						(KeyValuePair *)primitiveHash,
						(uint *)m_runOffsets->bufferOnDevice(),
                        m,
						d,
						numRuns);
	
	m_runLength->create(scanLength * 4);
	
	// CudaBase::CheckCudaError("write run hash");
#if COMPRESS_TWICE						
	sort(m_runHash->bufferOnDevice(), numRuns, 3*m);
	
    sahcompress::computeSortedRunLength((uint *)m_runLength->bufferOnDevice(),
							(uint *)m_runOffsets->bufferOnDevice(),
							(KeyValuePair *)m_runHash->bufferOnDevice(),
							numRuns,
							numPrimitives,
							scanLength);

	scanner()->prefixSum(m_runIndices, m_runLength, scanLength);

	// sahlg.writeUInt(m_runIndices, numRuns, "offset", CudaDbgLog::FOnce);

	sahdecompress::copyHash((KeyValuePair *)sortIntermediate(),
					(KeyValuePair *)primitiveHash,
					numPrimitives);
	
	sahdecompress::rearrangeIndices((KeyValuePair *)primitiveHash,
		    				(KeyValuePair *)sortIntermediate(),
		    				(uint *)m_runOffsets->bufferOnDevice(),
		    				(KeyValuePair *)m_runHash->bufferOnDevice(),
		    				(uint *)m_runIndices->bufferOnDevice(),
		    				(uint *)m_runLength->bufferOnDevice(),
		    				numRuns);
	
	// sahlg.writeHash(bvh->primitiveHashBuf(), numPrimitives, "primitive_hash", CudaDbgLog::FOnce);
 
	
	sahcompress::computeRunHead((uint *)m_runHeads->bufferOnDevice(), 
							(KeyValuePair *)primitiveHash,
							d,
							numPrimitives,
							scanLength);
		
	numRuns = scanner()->prefixSum(m_runIndices, 
									m_runHeads, scanLength);
												
	sahcompress::compressRunHead((uint *)m_runOffsets->bufferOnDevice(), 
							(uint *)m_runHeads->bufferOnDevice(),
							(uint *)m_runIndices->bufferOnDevice(),
							numPrimitives);
	
	sahdecompress::initHash((KeyValuePair *)m_runHash->bufferOnDevice(),
                             numRuns);

#endif 
    
	sahcompress::computeSortedRunLength((uint *)m_runLength->bufferOnDevice(),
							(uint *)m_runOffsets->bufferOnDevice(),
						    (KeyValuePair *)m_runHash->bufferOnDevice(),
							numRuns,
							numPrimitives,
							scanLength);
							
	// sahlg.writeUInt(m_runLength, numRuns, "run_length0", CudaDbgLog::FOnce);

    sahcompress::computeSortedClusterAabbs((Aabb *)clusterAabbs(),
            (KeyValuePair *)primitiveHash,
            (Aabb *)primitiveAabb,
            (KeyValuePair *)m_runHash->bufferOnDevice(),
            (uint *)m_runOffsets->bufferOnDevice(),
            (uint *)m_runLength->bufferOnDevice(),
            numRuns);
    
    return numRuns;
}

void * SahBuilder::clusterAabbs()
{ return m_clusterAabb->bufferOnDevice(); }
//:~
