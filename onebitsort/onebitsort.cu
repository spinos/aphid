/*
 *  one-block-one-bit counting sort
 */

#include "radixsort_implement.h"

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
 
__device__ void reduceInBlock(uint * binVertical)
{
        if(threadIdx.x < 128) {
        binVertical[0] += binVertical[0 + 128 *2];
        binVertical[1] += binVertical[1 + 128 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 64) {
        binVertical[0] += binVertical[0 + 64 *2];
        binVertical[1] += binVertical[1 + 64 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 32) {
        binVertical[0] += binVertical[0 + 32 *2];
        binVertical[1] += binVertical[1 + 32 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 16) {
        binVertical[0] += binVertical[0 + 16 *2];
        binVertical[1] += binVertical[1 + 16 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 8) {
        binVertical[0] += binVertical[0 + 8 *2];
        binVertical[1] += binVertical[1 + 8 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 4) {
        binVertical[0] += binVertical[0 + 4 *2];
        binVertical[1] += binVertical[1 + 4 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 2) {
        binVertical[0] += binVertical[0 + 2 *2];
        binVertical[1] += binVertical[1 + 2 *2];
    }
    __syncthreads();
    
    if(threadIdx.x < 1) {
        binVertical[0] += binVertical[0 + 1 *2];
        binVertical[1] += binVertical[1 + 1 *2];
    }
    __syncthreads();
}
 
__device__ void scanInBlock(uint * sum, uint* idata)
{
    int i = threadIdx.x;
// initial value
    sum[i*2] = idata[i*2];
    sum[i*2+1] = idata[i*2+1];
    __syncthreads();
    
// up sweep
    if((i & 1)==1) {
        sum[i*2] = sum[i*2] + sum[(i-1)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-1)*2+1];
    }
    __syncthreads();
    
    if((i & 3)==3) {
        sum[i*2] = sum[i*2] + sum[(i-2)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-2)*2+1];
    }
    __syncthreads();
    
    if((i & 7)==7) {
        sum[i*2] = sum[i*2] + sum[(i-4)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-4)*2+1];
    }
    __syncthreads();
    
    if((i & 15)==15) {
        sum[i*2] = sum[i*2] + sum[(i-8)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-8)*2+1];
    }
    __syncthreads();
    
    if((i & 31)==31) {
        sum[i*2] = sum[i*2] + sum[(i-16)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-16)*2+1];
    }
    __syncthreads();
    
    if((i & 63)==63) {
        sum[i*2] = sum[i*2] + sum[(i-32)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-32)*2+1];
    }
    __syncthreads();
    
    if((i & 127)==127) {
        sum[i*2] = sum[i*2] + sum[(i-64)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-64)*2+1];
    }
    __syncthreads();
    
    if(i==255) {
        sum[i*2] = sum[i*2] + sum[(i-128)*2];
        sum[i*2+1] = sum[i*2+1] + sum[(i-128)*2+1];
    }
    __syncthreads();
    
// down sweep
    uint tmp;
    if(i==255) {
        sum[i*2] = 0;
        sum[i*2+1] = 0;
        
        tmp = sum[(i-128)*2];
        sum[(i-128)*2] = 0;
        sum[i*2] = tmp;
        
        tmp = sum[(i-128)*2+1];
        sum[(i-128)*2+1] = 0;
        sum[i*2+1] = tmp;
    }
    __syncthreads();
    
    if((i & 127)==127) {
        tmp = sum[(i-64)*2];
        sum[(i-64)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-64)*2+1];
        sum[(i-64)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
    
    if((i & 63)==63) {
        tmp = sum[(i-32)*2];
        sum[(i-32)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-32)*2+1];
        sum[(i-32)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
    
    if((i & 31)==31) {
        tmp = sum[(i-16)*2];
        sum[(i-16)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-16)*2+1];
        sum[(i-16)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
    
    if((i & 15)==15) {
        tmp = sum[(i-8)*2];
        sum[(i-8)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-8)*2+1];
        sum[(i-8)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
    
    if((i & 7)==7) {
        tmp = sum[(i-4)*2];
        sum[(i-4)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-4)*2+1];
        sum[(i-4)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
    
    if((i & 3)==3) {
        tmp = sum[(i-2)*2];
        sum[(i-2)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-2)*2+1];
        sum[(i-2)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
    
    if((i & 1)==1) {
        tmp = sum[(i-1)*2];
        sum[(i-1)*2] = sum[i*2];
        sum[i*2] += tmp;
        
        tmp = sum[(i-1)*2+1];
        sum[(i-1)*2+1] = sum[i*2+1];
        sum[i*2+1] += tmp;
    }
    __syncthreads();
}
 
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
    
    reduceInBlock(binVertical);
    
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
    
    uint j, pos, p, ind;
    for(i=0;i<numBatches;i++) {
        for(j=0;j<2;j++) {
            binVertical[j] = 0;
            binOffsetVertical[j] = 0;
        }
        
        __syncthreads();
    
        pos = i*256+threadIdx.x;
            
        if(pos<elements) {
            p = pData[pos].key;
            binVertical[p]++;
        }
        
        __syncthreads();
        
#if 0
        if(threadIdx.x < 2) {
            for(j=1;j<256;j++) {
                binOffsetHorizontal[2 * j] = binOffsetHorizontal[2 * (j-1)] 
                                                + binHorizontal[2 * (j-1)];
            }
        }
        __syncthreads();
#else
        scanInBlock(&sRadixSum[4 + 256 * 2], &sRadixSum[4]);
#endif
        
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