#ifndef SAH_IMPLEMENT_H
#define SAH_IMPLEMENT_H

#include <bvh_common.h>
#include <radixsort_implement.h>

#define SAH_MAX_N_BLOCKS 2048
#define SIZE_OF_SPLITBIN 64
#define SIZE_OF_SPLITID 8

struct EmissionBlock {
    uint root_id;
    uint block_offset;
};

struct SplitBin {
    Aabb leftBox;
    uint leftCount;
    Aabb rightBox;
    uint rightCount;
    float cost;
    float plane;
};

struct SplitId {
    uint emissionId;
    uint side;
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

void sahbvh_emitSahSplit(EmissionBlock * outEmissions,
	    EmissionBlock * inEmissions,
	    int2 * rootNodes,
	    Aabb * rootAabbs,
	    KeyValuePair * clusterMorton,
        Aabb * clusterAabbs,
        SplitBin * splitBins,
        SplitId * splitIds,
	    uint numClusters,
        uint numBins,
	    uint numEmissions);

}
#endif        //  #ifndef SAH_IMPLEMENT_H

