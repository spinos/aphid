#include "cuReduceMinMaxBox_implement.h"

template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

template<class T>
__device__ void getMinMaxBox(T & a, T & b)
{
	a.low.x = min(a.low.x, b.low.x);
	a.low.y = min(a.low.y, b.low.y);
	a.low.z = min(a.low.z, b.low.z);
	a.high.x = max(a.high.x, b.high.x);
	a.high.y = max(a.high.y, b.high.y);
	a.high.z = max(a.high.z, b.high.z);
}

template<class T, class T1>
__device__ void getMinMaxBox(T & a, T1 & b)
{
	a.low.x = min(a.low.x, b.x);
	a.low.y = min(a.low.y, b.y);
	a.low.z = min(a.low.z, b.z);
	a.high.x = max(a.high.x, b.x);
	a.high.y = max(a.high.y, b.y);
	a.high.z = max(a.high.z, b.z);
}

template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void reduceFindMinMaxBox_kernel(T *g_idata, T *g_odata, unsigned int n)
{
	T *sdata = SharedMemory<T>();
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum;
    mySum.low.x = 999999999;
	mySum.low.y = 999999999;
	mySum.low.z = 999999999;
    mySum.high.x = -999999999;
	mySum.high.y = -999999999;
	mySum.high.z = -999999999;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
		getMinMaxBox<T>(mySum, g_idata[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
			getMinMaxBox<T>(mySum, g_idata[i+blockSize]);
        }
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512 && tid < 256) { 
			getMinMaxBox<T>(mySum, sdata[tid + 256]);
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 
    
    if (blockSize >= 256 && tid < 128) {
			getMinMaxBox<T>(mySum, sdata[tid + 128]);
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 
   
    if (blockSize >= 128 && tid <  64) { 
			getMinMaxBox<T>(mySum, sdata[tid + 64]);
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 

        // fully unroll reduction within a single warp
        if ((blockSize >=  64) && (tid < 32)) {
			getMinMaxBox<T>(mySum, sdata[tid + 32]);
            sdata[tid] = mySum;
        }
	 __syncthreads(); 
	
        if ((blockSize >=  32) && (tid < 16)) { 
            getMinMaxBox<T>(mySum, sdata[tid + 16]);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=  16) && (tid <  8)) { 
            getMinMaxBox<T>(mySum, sdata[tid + 8]);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=   8) && (tid <  4)) {
            getMinMaxBox<T>(mySum, sdata[tid + 4]);
            sdata[tid] = mySum;
        }
	 __syncthreads(); 
	
       if ((blockSize >=   4) && (tid <  2)) { 
            getMinMaxBox<T>(mySum, sdata[tid + 2]);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=   2) && ( tid <  1)) { 
           getMinMaxBox<T>(mySum, sdata[tid + 1]);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <class T, class T1, unsigned int blockSize, bool nIsPow2>
__global__ void reduceFindMinMaxBox_kernel1(T1 *g_idata, T *g_odata, unsigned int n)
{
	T *sdata = SharedMemory<T>();
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum;
    mySum.low.x = 999999999;
	mySum.low.y = 999999999;
	mySum.low.z = 999999999;
    mySum.high.x = -999999999;
	mySum.high.y = -999999999;
	mySum.high.z = -999999999;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
		getMinMaxBox<T, T1>(mySum, g_idata[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
			getMinMaxBox<T, T1>(mySum, g_idata[i+blockSize]);
        }
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512 && tid < 256) { 
			getMinMaxBox<T>(mySum, sdata[tid + 256]);
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 
    
    if (blockSize >= 256 && tid < 128) {
			getMinMaxBox<T>(mySum, sdata[tid + 128]);
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 
   
    if (blockSize >= 128 && tid <  64) { 
			getMinMaxBox<T>(mySum, sdata[tid + 64]);
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 

        // fully unroll reduction within a single warp
        if ((blockSize >=  64) && (tid < 32)) {
			getMinMaxBox<T>(mySum, sdata[tid + 32]);
            sdata[tid] = mySum;
        }
	 __syncthreads(); 
	
        if ((blockSize >=  32) && (tid < 16)) { 
            getMinMaxBox<T>(mySum, sdata[tid + 16]);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=  16) && (tid <  8)) { 
            getMinMaxBox<T>(mySum, sdata[tid + 8]);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=   8) && (tid <  4)) {
            getMinMaxBox<T>(mySum, sdata[tid + 4]);
            sdata[tid] = mySum;
        }
	 __syncthreads(); 
	
       if ((blockSize >=   4) && (tid <  2)) { 
            getMinMaxBox<T>(mySum, sdata[tid + 2]);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=   2) && ( tid <  1)) { 
           getMinMaxBox<T>(mySum, sdata[tid + 1]);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <class T>
void cuReduceFindMinMaxBox(T *dst, T *src, unsigned numElements, unsigned numBlocks, unsigned numThreads)
{
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
	int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
	
	if (isPow2(numElements)) {
		switch (numThreads)
		{
		case 512:
			reduceFindMinMaxBox_kernel<T,512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 256:
			reduceFindMinMaxBox_kernel<T,256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 128:
			reduceFindMinMaxBox_kernel<T,128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 64:
			reduceFindMinMaxBox_kernel<T,64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 32:
			reduceFindMinMaxBox_kernel<T,32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 16:
			reduceFindMinMaxBox_kernel<T,16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  8:
			reduceFindMinMaxBox_kernel<T, 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  4:
			reduceFindMinMaxBox_kernel<T, 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  2:
			reduceFindMinMaxBox_kernel<T, 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  1:
			reduceFindMinMaxBox_kernel<T, 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		}
	}
	else {
		switch (numThreads)
		{
		case 512:
			reduceFindMinMaxBox_kernel<T,512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 256:
			reduceFindMinMaxBox_kernel<T,256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 128:
			reduceFindMinMaxBox_kernel<T,128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 64:
			reduceFindMinMaxBox_kernel<T,64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 32:
			reduceFindMinMaxBox_kernel<T,32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 16:
			reduceFindMinMaxBox_kernel<T,16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  8:
			reduceFindMinMaxBox_kernel<T, 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  4:
			reduceFindMinMaxBox_kernel<T, 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  2:
			reduceFindMinMaxBox_kernel<T, 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  1:
			reduceFindMinMaxBox_kernel<T, 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		}
	}
}

template void 
cuReduceFindMinMaxBox<Aabb>(Aabb *d_odata, Aabb *d_idata, 
    unsigned numElements, unsigned numBlocks, unsigned numThreads);

template <class T, class T1>
void cuReduceFindMinMaxBox(T *dst, T1 *src, unsigned numElements, unsigned numBlocks, unsigned numThreads)
{
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
	int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
	
	if (isPow2(numElements)) {
		switch (numThreads)
		{
		case 512:
			reduceFindMinMaxBox_kernel1<T, T1, 512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 256:
			reduceFindMinMaxBox_kernel1<T, T1, 256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 128:
			reduceFindMinMaxBox_kernel1<T, T1, 128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 64:
			reduceFindMinMaxBox_kernel1<T, T1, 64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 32:
			reduceFindMinMaxBox_kernel1<T, T1, 32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 16:
			reduceFindMinMaxBox_kernel1<T, T1, 16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  8:
			reduceFindMinMaxBox_kernel1<T, T1,  8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  4:
			reduceFindMinMaxBox_kernel1<T, T1,  4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  2:
			reduceFindMinMaxBox_kernel1<T, T1,  2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  1:
			reduceFindMinMaxBox_kernel1<T, T1,  1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		}
	}
	else {
		switch (numThreads)
		{
		case 512:
			reduceFindMinMaxBox_kernel1<T, T1, 512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 256:
			reduceFindMinMaxBox_kernel1<T, T1, 256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 128:
			reduceFindMinMaxBox_kernel1<T, T1, 128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 64:
			reduceFindMinMaxBox_kernel1<T, T1, 64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 32:
			reduceFindMinMaxBox_kernel1<T, T1, 32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 16:
			reduceFindMinMaxBox_kernel1<T, T1, 16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  8:
			reduceFindMinMaxBox_kernel1<T, T1,  8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  4:
			reduceFindMinMaxBox_kernel1<T, T1,  4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  2:
			reduceFindMinMaxBox_kernel1<T, T1,  2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  1:
			reduceFindMinMaxBox_kernel1<T, T1,  1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		}
	}
}

template void 
cuReduceFindMinMaxBox<Aabb, float3>(Aabb *d_odata, float3 *d_idata, 
    unsigned numElements, unsigned numBlocks, unsigned numThreads);
