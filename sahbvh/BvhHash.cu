#include "BvhHash.cuh"

namespace bvhhash {
    
void computePrimitiveHash(KeyValuePair * dst, Aabb * leafBoxes, uint numLeaves, uint buffSize, 
			Aabb * bigBox)
{
	dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(buffSize, 512);
    
    dim3 grid(nblk, 1, 1);
	calculateLeafHash_kernel<<< grid, block >>>(dst, leafBoxes, numLeaves, buffSize, bigBox);
}

}
