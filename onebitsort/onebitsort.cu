/*
 *  one-block-one-bit counting sort
 */
#include "onebitsort.cuh"

extern __shared__ uint sRadixSum[];

/*
 *  shared memory layout
 *  0 -> 1               group offset
 *  2 -> 3               group count
 *  4 -> 4+n*2-1         per-thread count
 *  4+n*2 -> 4+n*2+n*2-1 per-thread offset
 *
 *  per-thread count layout
 *  thread count: n
 *  bin count   : 2
 * 
 *  0      1      2        n-1     thread 
 * 
 *  2*0    2*1    2*2      2*(n-1)
 *  2*1-1  2*2-1  2*3-1    2*n-1
 *
 *  prefix sum: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
 */
 
__global__ void RadixSum(KeyValuePair *oData, KeyValuePair *pData, 
                        uint elements, 
        uint * counts)
{
    uint * binVertical = &sRadixSum[4 + threadIdx.x * 2];
    uint * binHorizontal = &sRadixSum[4 + threadIdx.x];
    uint * binOffsetVertical = &sRadixSum[4 + 256 * 2 + threadIdx.x * 2];
    uint * binOffsetHorizontal = &sRadixSum[4 + 256 * 2 + threadIdx.x];
    
    uint i = 0;
    for(;i<2;i++)
        binVertical[i] = 0;
    
    uint numBatches = elements>>8;
    if(elements & 255) numBatches++;
    
    for(i=0;i<numBatches;i++) {
        uint pos = i*256+threadIdx.x;
        if(pos<elements) {
            uint p = pData[pos].key;
            binVertical[p]++;
        }
    }
    
    __syncthreads();
    
    onebitsort::reduceInBlock(binVertical);
    
    if(threadIdx.x < 4)
        sRadixSum[threadIdx.x] = 0;
    
    __syncthreads();
    
    if(threadIdx.x < 2) {
        sRadixSum[2 + threadIdx.x] = binHorizontal[0];
        counts[2 + threadIdx.x] = sRadixSum[2 + threadIdx.x];
    }
    
    __syncthreads();
    
    if(threadIdx.x == 1) {
        sRadixSum[threadIdx.x] += sRadixSum[2 + threadIdx.x - 1];
    }
    
    if(threadIdx.x < 2) {
        counts[threadIdx.x] = sRadixSum[threadIdx.x];
    }

    __syncthreads();
    
    uint pos, p, ind;
    for(i=0;i<numBatches;i++) {
        binVertical[0] = 0;
        binVertical[1] = 0;
        
        __syncthreads();
    
        pos = i*256+threadIdx.x;
            
        if(pos<elements) {
            p = pData[pos].key;
            binVertical[p]++;
        }
        
        __syncthreads();
        
        onebitsort::scanInBlock(&sRadixSum[4 + 256 * 2], &sRadixSum[4]);
        
        if(pos<elements) {
            ind = sRadixSum[p] + binOffsetVertical[p];
            oData[ind] = pData[pos];
        }
        
        __syncthreads();
        
        if(threadIdx.x < 2) {
            sRadixSum[threadIdx.x] += binOffsetHorizontal[2*255]
                                        + binHorizontal[2*255];
        }
        
        __syncthreads();
    }
}

#include <stdio.h>

void OneBitSort(KeyValuePair *pData0, KeyValuePair *pData1, uint elements, uint * counts)
{
    RadixSum<<<1, 256, 16320>>>(pData1, pData0, elements, counts);
    
    uint hbins[4] = {0,0,0,0};
    cudaError_t err = cudaMemcpy(hbins, counts, 16, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        printf(" error group counts %s\n", cudaGetErrorString(err));
    
    printf(" offset0 %i \n", hbins[0]);
    printf(" offset1 %i \n", hbins[1]);
    printf(" count0 %i \n", hbins[2]);
    printf(" count1 %i \n", hbins[3]);
}