/*
 *  SahBuilder.cpp
 *  testsah
 *
 *  Created by jian zhang on 5/4/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
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
}

SahBuilder::~SahBuilder() 
{
	delete m_runHeads;
	delete m_runIndices;
	delete m_runHash;
	delete m_runLength;
	delete m_compressedRunHeads;
}

void SahBuilder::initOnDevice()
{
	BvhBuilder::initOnDevice();
}

void SahBuilder::build(CudaLinearBvh * bvh)
{
// create data
	const unsigned nl = bvh->numPrimitives();
	createSortAndScanBuf(nl);
	
	m_runHeads->create(CudaScan::getScanBufferLength(nl) * 4);
	m_runIndices->create(CudaScan::getScanBufferLength(nl) * 4);
	
	computeMortionHash(bvh->leafHash(), bvh->leafAabbs(), nl);
	
	sortPrimitives(bvh->leafHash(), nl, 5);
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
