#include "reduceRange_implement.h"

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceFindMax_kernel(int *g_idata, int *g_odata, unsigned int n)
{
    extern __shared__ int sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    int mySum = -9999999;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        mySum = max(mySum, g_idata[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            mySum = max(mySum, g_idata[i+blockSize]);  
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { 
        if (tid < 256) { 
            mySum = max(mySum, sdata[tid + 256]);
            sdata[tid] = mySum; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            mySum = max(mySum, sdata[tid + 128]); 
            sdata[tid] = mySum; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            mySum = max(mySum, sdata[tid +  64]); 
            sdata[tid] = mySum; 
        } 
        __syncthreads(); 
    }
    
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile int * smem = sdata;
        if (blockSize >=  64) {
            mySum = max(mySum, smem[tid + 32]);
            smem[tid] = mySum;
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
            mySum = max(mySum, smem[tid + 16]);
            smem[tid] = mySum;
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            mySum = max(mySum, smem[tid +  8]);
            smem[tid] = mySum;
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            mySum = max(mySum, smem[tid +  4]);
            smem[tid] = mySum;
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            mySum = max(mySum, smem[tid +  2]);
            smem[tid] = mySum;
            __syncthreads(); 
        }
        if (blockSize >=   2) { 
            mySum = max(mySum, smem[tid +  1]);
            smem[tid] = mySum;
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

extern "C" void bvhReduceFindMax(int *dst, int *src, unsigned numElements, unsigned numBlocks, unsigned numThreads)
{
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
	int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(int) : numThreads * sizeof(int);
	
	if (isPow2(numElements)) {
		switch (numThreads)
		{
		case 512:
			reduceFindMax_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 256:
			reduceFindMax_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 128:
			reduceFindMax_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 64:
			reduceFindMax_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 32:
			reduceFindMax_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 16:
			reduceFindMax_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  8:
			reduceFindMax_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  4:
			reduceFindMax_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  2:
			reduceFindMax_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  1:
			reduceFindMax_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		}
	}
	else {
		switch (numThreads)
		{
		case 512:
			reduceFindMax_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 256:
			reduceFindMax_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 128:
			reduceFindMax_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 64:
			reduceFindMax_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 32:
			reduceFindMax_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 16:
			reduceFindMax_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  8:
			reduceFindMax_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  4:
			reduceFindMax_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  2:
			reduceFindMax_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  1:
			reduceFindMax_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		}
	}
}
