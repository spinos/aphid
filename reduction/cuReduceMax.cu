#include "cuReduceMax_implement.h"

template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduceFindMax_kernel(T *g_idata, T *g_odata, unsigned int n)
{
	T *sdata = SharedMemory<T>();
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum = -999999999;

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
    if (blockSize >= 512 && tid < 256) { 
            mySum = max(mySum, sdata[tid + 256]);
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 
    
    if (blockSize >= 256 && tid < 128) { 
            mySum = max(mySum, sdata[tid + 128]); 
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 
   
    if (blockSize >= 128 && tid <  64) { 
            mySum = max(mySum, sdata[tid +  64]); 
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 
 
        // fully unroll reduction within a single warp
        if ((blockSize >=  64) && (tid < 32)) {
            mySum = max(mySum, sdata[tid + 32]);
            sdata[tid] = mySum;
        }
	 __syncthreads(); 
	
        if ((blockSize >=  32) && (tid < 16)) { 
            mySum = max(mySum, sdata[tid + 16]);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=  16) && (tid <  8)) { 
            mySum = max(mySum, sdata[tid +  8]);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=   8) && (tid <  4)) { 
            mySum = max(mySum, sdata[tid +  4]);
            sdata[tid] = mySum;
        }
	 __syncthreads(); 
	
       if ((blockSize >=   4) && (tid <  2)) { 
            mySum = max(mySum, sdata[tid +  2]);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=   2) && ( tid <  1)) { 
            mySum = max(mySum, sdata[tid +  1]);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <class T>
void cuReduceFindMax(T *dst, T *src, unsigned numElements, unsigned numBlocks, unsigned numThreads)
{
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
	int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
	
	if (isPow2(numElements)) {
		switch (numThreads)
		{
		case 512:
			reduceFindMax_kernel<T,512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 256:
			reduceFindMax_kernel<T,256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 128:
			reduceFindMax_kernel<T,128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 64:
			reduceFindMax_kernel<T,64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 32:
			reduceFindMax_kernel<T,32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 16:
			reduceFindMax_kernel<T,16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  8:
			reduceFindMax_kernel<T, 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  4:
			reduceFindMax_kernel<T, 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  2:
			reduceFindMax_kernel<T, 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  1:
			reduceFindMax_kernel<T, 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		}
	}
	else {
		switch (numThreads)
		{
		case 512:
			reduceFindMax_kernel<T,512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 256:
			reduceFindMax_kernel<T,256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 128:
			reduceFindMax_kernel<T,128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 64:
			reduceFindMax_kernel<T,64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 32:
			reduceFindMax_kernel<T,32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 16:
			reduceFindMax_kernel<T,16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  8:
			reduceFindMax_kernel<T, 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  4:
			reduceFindMax_kernel<T, 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  2:
			reduceFindMax_kernel<T, 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  1:
			reduceFindMax_kernel<T, 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		}
	}
}

template void 
cuReduceFindMax<int>(int *d_idata, int *d_odata, 
    unsigned numElements, unsigned numBlocks, unsigned numThreads);

template void 
cuReduceFindMax<float>(float *d_idata, float *d_odata,
    unsigned numElements, unsigned numBlocks, unsigned numThreads);

