#ifndef SAH_IMPLEMENT_H
#define SAH_IMPLEMENT_H

#include <bvh_common.h>
#include <radixsort_implement.h>

#define SAH_MAX_N_BLOCKS 4096

struct EmissionBlock {
    uint root_id;
    uint block_offset;
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

}
#endif        //  #ifndef SAH_IMPLEMENT_H

