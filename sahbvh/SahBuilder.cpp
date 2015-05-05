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
#include <sahbvh_implement.h>
#include <CudaBase.h>
#include <CudaScan.h>

CudaDbgLog sahlg("sah.txt");

SahBuilder::SahBuilder() 
{
	m_runHeads = new CUDABuffer;
	m_runIndices = new CUDABuffer;
	m_runHash = new CUDABuffer;
	m_runLength = new CUDABuffer;
	m_compressedRunHeads = new CUDABuffer;
	m_emissions[0] = new CUDABuffer;
	m_emissions[1] = new CUDABuffer;
}

SahBuilder::~SahBuilder() 
{
	delete m_runHeads;
	delete m_runIndices;
	delete m_runHash;
	delete m_runLength;
	delete m_compressedRunHeads;
	delete m_emissions[0];
	delete m_emissions[1];
}

void SahBuilder::initOnDevice()
{
    m_emissions[0]->create(SAH_MAX_N_BLOCKS * 8);
    m_emissions[1]->create(SAH_MAX_N_BLOCKS * 8);
	BvhBuilder::initOnDevice();
}

void SahBuilder::build(CudaLinearBvh * bvh)
{
// create data
	const unsigned n = bvh->numPrimitives();
	createSortAndScanBuf(n);
	
	m_runHeads->create(CudaScan::getScanBufferLength(n) * 4);
	m_runIndices->create(CudaScan::getScanBufferLength(n) * 4);
	
	void * primitiveHash = bvh->primitiveHash();
	void * primitiveAabb = bvh->primitiveAabb();
	
	computeMortionHash(primitiveHash, primitiveAabb, n);
	
	if(n<=2048) sort(primitiveHash, n, 32);
	else sortPrimitives(primitiveHash, n, 6);
	
// set root node range
    unsigned rr[2];
    rr[0] = 0;
    rr[1] = n - 1;
    
    CUDABuffer * internalChildBuf = bvh->internalNodeChildIndicesBuf();
    internalChildBuf->hostToDevice(rr, 8);
	
// emit from root node
	EmissionBlock eb;
	eb.root_id = 0;
	eb.block_offset = 0;
	
	m_emissions[0]->hostToDevice(&eb, 8);
	unsigned nEmissions = 1;
	
	
}

void SahBuilder::sortPrimitives(void * morton, unsigned numPrimitives, unsigned m)
{
	const unsigned scanLength = CudaScan::getScanBufferLength(numPrimitives);
	
	const unsigned d = 30 - 3*m;
	sahbvh_computeRunHead((uint *)m_runHeads->bufferOnDevice(), 
							(KeyValuePair *)morton,
							d,
							numPrimitives,
							scanLength);
	
	CudaBase::CheckCudaError("finding bvh run heads");
	
	sahlg.writeUInt(m_runHeads, numPrimitives, "run_heads", CudaDbgLog::FOnce);
			
	const unsigned numRuns = scanner()->prefixSum(m_runIndices, 
												m_runHeads, scanLength);
												
	CudaBase::CheckCudaError("scan bvh run heads");
	
	sahlg.writeUInt(m_runIndices, numPrimitives, "scanned_run_heads", CudaDbgLog::FOnce);
	
// no need to compress												
	if(numRuns == numPrimitives - 1) {
		sort(morton, numPrimitives, 32);
		return;
	}
	
	m_compressedRunHeads->create(numRuns * 4);
	
	sahbvh_compressRunHead((uint *)m_compressedRunHeads->bufferOnDevice(), 
							(uint *)m_runHeads->bufferOnDevice(),
							(uint *)m_runIndices->bufferOnDevice(),
							numPrimitives);
							
	sahlg.writeUInt(m_compressedRunHeads, numRuns, "compressed_run_heads", CudaDbgLog::FOnce);
	
	const unsigned sortRunLength = nextPow2(numRuns);
	m_runHash->create((sortRunLength * sizeof(KeyValuePair)));
	
	sahbvh_computeRunHash((KeyValuePair *)m_runHash->bufferOnDevice(), 
						(KeyValuePair *)morton,
						(uint *)m_runIndices->bufferOnDevice(),
						d,
						numRuns,
						sortRunLength);
	
	CudaBase::CheckCudaError("write run hash");
						
	sort(m_runHash->bufferOnDevice(), numRuns, 25);
	
	sahlg.writeHash(m_runHash, numRuns, "sorted_run_heads", CudaDbgLog::FOnce);

	m_runLength->create(scanLength * 4);
	
	sahbvh_computeRunLength((uint *)m_runLength->bufferOnDevice(),
							(uint *)m_compressedRunHeads->bufferOnDevice(),
							(KeyValuePair *)m_runHash->bufferOnDevice(),
							numRuns,
							numPrimitives,
							scanLength);
							
	sahlg.writeUInt(m_runLength, numRuns, "run_length", CudaDbgLog::FOnce);

	scanner()->prefixSum(m_runIndices, m_runLength, scanLength);
	
	sahlg.writeUInt(m_runIndices, numRuns, "offset", CudaDbgLog::FOnce);

	sahbvh_copyHash((KeyValuePair *)sortIntermediate(),
					(KeyValuePair *)morton,
					numPrimitives);
					
	sahbvh_decompressIndices((uint *)m_runHeads->bufferOnDevice(),
					(uint *)m_compressedRunHeads->bufferOnDevice(),
					(KeyValuePair *)m_runHash->bufferOnDevice(),
					(uint *)m_runIndices->bufferOnDevice(),
					(uint *)m_runLength->bufferOnDevice(),
					numRuns);
					
	sahlg.writeUInt(m_runHeads, numPrimitives, "decompressed_ind", CudaDbgLog::FOnce);
	
	sahbvh_writeSortedHash((KeyValuePair *)morton,
							(KeyValuePair *)sortIntermediate(),
							(uint *)m_runHeads->bufferOnDevice(),
							numPrimitives);
}
