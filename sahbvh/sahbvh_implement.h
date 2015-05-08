#ifndef SAH_IMPLEMENT_H
#define SAH_IMPLEMENT_H

#include <bvh_common.h>
#include <radixsort_implement.h>

#define SAH_MAX_NUM_BINS 16
#define SAH_MAX_N_BLOCKS 2048
#define SIZE_OF_SPLITBIN 64
#define SIZE_OF_SPLITID 8
#define SIZE_OF_EMISSIONBLOCK 8
#define SIZE_OF_EMISSIONEVENT 16

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

struct SplitBin {
    BinAabb leftBox;
    uint leftCount;
    BinAabb rightBox;
    uint rightCount;
    float cost;
    float plane;
};

struct EmissionBlock {
    uint emission_id;
    uint primitive_offset;
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
						uint n,
						uint bufLength);

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
        uint * totalBinningBlocks,
        uint * totalNodeCount,
	    uint numClusters,
        uint numBins,
	    uint numEmissions,
	    uint currentNumNodes);

}
#endif        //  #ifndef SAH_IMPLEMENT_H

