#ifndef SAH_IMPLEMENT_H
#define SAH_IMPLEMENT_H

#include "sah_common.h"

#define SAH_MAX_NUM_BINS 16
#define SAH_MAX_N_BLOCKS 1024
#define SIZE_OF_SPLITBIN 64
#define SIZE_OF_SPLITID 8
#define SIZE_OF_EMISSIONBLOCK 16
#define SIZE_OF_EMISSIONEVENT 16
#define COMPUTE_BINS_NTHREAD 128
#define COMPUTE_BINS_NTHREAD_M1 127
#define COMPUTE_BINS_NTHREAD_LOG2 7

struct EmissionEvent {
    uint root_id;
    uint node_offset;
    int bin_id;
    int n_split;
};

struct BinAabb {
    int3 low;
    int3 high;
};

struct EmissionBlock {
    uint emission_id;
    uint primitive_offset;
    int is_spilled;
    int bin_offset;
};

struct SplitId {
    uint emission_id;
    uint split_side;
};

extern "C" {
void sahbvh_computeRunHead(uint * blockHeads, 
							KeyValuePair * mortonCode,
							uint d,
							uint n,
							uint bufLength);
							
void sahbvh_computeRunHash(KeyValuePair * compressed, 
						KeyValuePair * morton,
						uint * indices,
                        uint m,
						uint d,
						uint n);

void sahbvh_computeRunLength(uint * runLength,
							uint * runHeads,
							KeyValuePair * indices,
							uint nRuns,
							uint nPrimitives,
							uint bufLength);

void sahbvh_compressRunHead(uint * compressed, 
							uint * runHeads,
							uint * indices,
							uint n);

void sahbvh_copyHash(KeyValuePair * dst,
					KeyValuePair * src,
					uint n);

void sahbvh_decompressIndices(uint * decompressedIndices,
                    uint * compressedIndices,
					KeyValuePair * sorted,
					uint * offset,
					uint * runLength,
					uint n);

void sahbvh_computeClusterAabbs(Aabb * clusterAabbs,
            Aabb * primitiveAabbs,
            uint * compressedIndices,
            KeyValuePair * sorted,
            uint * offset,
            uint * runLength,
            uint n);

void sahbvh_writeSortedHash(KeyValuePair * dst,
							KeyValuePair * src,
							uint * indices,
							uint n);

void sahbvh_countTreeBits(uint * nbits, 
                            KeyValuePair * morton,
                            uint n);

void sahbvh_emitSahSplit(EmissionEvent * outEmissions,
	    EmissionEvent * inEmissions,
	    int2 * rootNodes,
	    Aabb * rootAabbs,
	    KeyValuePair * clusterIndirection,
        Aabb * clusterAabbs,
        SplitBin * splitBins,
        EmissionBlock * emissionIds,
        SplitId * splitIds,
        uint numBinningBlocks,
        uint numSpilledBinBlocks,
        uint * totalNodeCount,
	    uint numClusters,
        uint numBins,
	    uint numEmissions,
	    uint currentNumNodes);

}
#endif        //  #ifndef SAH_IMPLEMENT_H

