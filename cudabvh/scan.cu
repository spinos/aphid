/*
#include <cuda_runtime.h>


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/detail/type_traits.h>

#include <cutil_inline.h>

#include <shrUtils.h>
#include <algorithm>
#include <time.h>
#include <limits.h>
*/
#include "scan_implement.h"


#define THREADBLOCK_SIZE 256
extern "C" const uint MAX_BATCH_ELEMENTS = 64 * 1048576;
extern "C" const uint MIN_SHORT_ARRAY_SIZE = 4;
extern "C" const uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;


static uint factorRadix2(uint& log2L, uint L){
    if(!L){
        log2L = 0;
        return 0;
    }else{
        for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
        return L;
    }
}



#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

    //Almost the same as naive scan1Inclusive, but doesn't need __syncthreads()
    //assuming size <= WARP_SIZE
    inline __device__ uint warpScanInclusive(uint idata, volatile uint *s_Data, uint size){
        uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
        s_Data[pos] = 0;
        pos += size;
        s_Data[pos] = idata;

        for(uint offset = 1; offset < size; offset <<= 1)
            s_Data[pos] += s_Data[pos - offset];

        return s_Data[pos];
    }

    inline __device__ uint warpScanExclusive(uint idata, volatile uint *s_Data, uint size){
        return warpScanInclusive(idata, s_Data, size) - idata;
    }

inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size){
        if(size > WARP_SIZE){
            //Bottom-level inclusive warp scan
            uint warpResult = warpScanInclusive(idata, s_Data, WARP_SIZE);

            //Save top elements of each warp for exclusive warp scan
            //sync to wait for warp scans to complete (because s_Data is being overwritten)
            __syncthreads();
            if( (threadIdx.x & (WARP_SIZE - 1)) == (WARP_SIZE - 1) )
                s_Data[threadIdx.x >> LOG2_WARP_SIZE] = warpResult;

            //wait for warp scans to complete
            __syncthreads();
            if( threadIdx.x < (THREADBLOCK_SIZE / WARP_SIZE) ){
                //grab top warp elements
                uint val = s_Data[threadIdx.x];
                //calculate exclsive scan and write back to shared memory
                s_Data[threadIdx.x] = warpScanExclusive(val, s_Data, size >> LOG2_WARP_SIZE);
            }

            //return updated warp scans with exclusive scan results
            __syncthreads();
            return warpResult + s_Data[threadIdx.x >> LOG2_WARP_SIZE];
        }else{
            return warpScanInclusive(idata, s_Data, size);
        }
    }

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size){
        return scan1Inclusive(idata, s_Data, size) - idata;
    }

inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size){
    //Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    //Level-1 exclusive scan
    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}

inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size){
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

__global__ void scanExclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint size
){
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    uint4 idata4 = d_Src[pos];

    //Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, size);

    //Write back
    d_Dst[pos] = odata4;
}

//Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanExclusiveShared2(
    uint *d_Buf,
    uint *d_Dst,
    uint *d_Src,
    uint N,
    uint arrayLength
){
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    //Skip loads and stores for inactive threads of last threadblock (pos >= N)
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    uint idata = 0;
    if(pos < N)
        idata = 
        d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos] + 
        d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];

    //Compute
    uint odata = scan1Exclusive(idata, s_Data, arrayLength);

    //Avoid out-of-bound access
    if(pos < N)
        d_Buf[pos] = odata;
}

__global__ void
mergeScanGroups(uint *in, uint *element, uint *out)
{
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;
    uint offset = 0;
    uint i_group = pos/1024;
    for(uint i = 0; i < i_group; i++)
    {
        offset += in[i * 1024 + 1023] + element[i * 1024 + 1023];
        
    }
    out[pos] = in[pos] + offset;
    
}

extern "C" void scanExclusive(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength
)
{
    //Check power-of-two factorization
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    assert( factorizationRemainder == 1 );

    //Check supported size range
    assert( (arrayLength >= MIN_SHORT_ARRAY_SIZE) && (arrayLength <= MAX_SHORT_ARRAY_SIZE) );

    //Check total batch size limit
    assert( (batchSize * arrayLength) <= MAX_BATCH_ELEMENTS );

    //Check all threadblocks to be fully packed with data
    assert( (batchSize * arrayLength) % (4 * THREADBLOCK_SIZE) == 0 );

    uint *d_intermediate;
    // cutilSafeCall( 
        cudaMalloc((void **)&d_intermediate, sizeof(uint) * (batchSize * arrayLength) );
        //);
    
    scanExclusiveShared<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_intermediate,
        (uint4 *)d_Src,
        arrayLength
    );
    
    mergeScanGroups<<<(batchSize * arrayLength)/32, 32>>>(d_intermediate, d_Src, d_Dst);
    
    
    // cutilSafeCall( 
    cudaFree(d_intermediate);
    //);
}
