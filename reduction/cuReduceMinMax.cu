#include "cuReduceMinMax_implement.h"

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
__global__ void reduceFindMinMax_kernel(T *g_idata, T *g_odata, unsigned int n)
{
	T *sdata = SharedMemory<T>();
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum;
    mySum.x = 999999999;
    mySum.y = -999999999;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        mySum.x = min(mySum.x, g_idata[i].x);
        mySum.y = max(mySum.y, g_idata[i].y);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            mySum.x = min(mySum.x, g_idata[i+blockSize].x);  
            mySum.y = max(mySum.y, g_idata[i+blockSize].y);
        }
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512 && tid < 256) { 
            mySum.x = min(mySum.x, sdata[tid + 256].x);
            mySum.y = max(mySum.y, sdata[tid + 256].y);
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 
    
    if (blockSize >= 256 && tid < 128) { 
            mySum.x = min(mySum.x, sdata[tid + 128].x); 
            mySum.y = max(mySum.y, sdata[tid + 128].y); 
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 
   
    if (blockSize >= 128 && tid <  64) { 
            mySum.x = min(mySum.x, sdata[tid +  64].x); 
            mySum.y = max(mySum.y, sdata[tid +  64].y); 
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 

 #if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else 
        // fully unroll reduction within a single warp
        if ((blockSize >=  64) && (tid < 32)) {
            mySum.x = min(mySum.x, sdata[tid + 32].x);
            mySum.y = max(mySum.y, sdata[tid + 32].y);
            sdata[tid] = mySum;
        }
	 __syncthreads(); 
	
        if ((blockSize >=  32) && (tid < 16)) { 
            mySum.x = min(mySum.x, sdata[tid + 16].x);
            mySum.y = max(mySum.y, sdata[tid + 16].y);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=  16) && (tid <  8)) { 
            mySum.x = min(mySum.x, sdata[tid +  8].x);
            mySum.y = max(mySum.y, sdata[tid +  8].y);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=   8) && (tid <  4)) {
            mySum.x = min(mySum.x, sdata[tid +  4].x);
            mySum.y = max(mySum.y, sdata[tid +  4].y);
            sdata[tid] = mySum;
        }
	 __syncthreads(); 
	
       if ((blockSize >=   4) && (tid <  2)) { 
            mySum.x = min(mySum.x, sdata[tid +  2].x);
            mySum.y = max(mySum.y, sdata[tid +  2].y);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=   2) && ( tid <  1)) { 
            mySum.x = min(mySum.x, sdata[tid +  1].x);
            mySum.y = max(mySum.y, sdata[tid +  1].y);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
#endif
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <class T, class T1, unsigned int blockSize, bool nIsPow2>
__global__ void reduceFindMinMax_kernel1(T1 *g_idata, T *g_odata, unsigned int n)
{
	T *sdata = SharedMemory<T>();
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    T mySum;
    mySum.x = 999999999;
    mySum.y = -999999999;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        mySum.x = min(mySum.x, g_idata[i]);
        mySum.y = max(mySum.y, g_idata[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            mySum.x = min(mySum.x, g_idata[i+blockSize]);  
            mySum.y = max(mySum.y, g_idata[i+blockSize]);
        }
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512 && tid < 256) { 
            mySum.x = min(mySum.x, sdata[tid + 256].x);
            mySum.y = max(mySum.y, sdata[tid + 256].y);
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 
    
    if (blockSize >= 256 && tid < 128) { 
            mySum.x = min(mySum.x, sdata[tid + 128].x); 
            mySum.y = max(mySum.y, sdata[tid + 128].y); 
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 
   
    if (blockSize >= 128 && tid <  64) { 
            mySum.x = min(mySum.x, sdata[tid +  64].x); 
            mySum.y = max(mySum.y, sdata[tid +  64].y); 
            sdata[tid] = mySum; 
        } 

        __syncthreads(); 

 #if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else 
        // fully unroll reduction within a single warp
        if ((blockSize >=  64) && (tid < 32)) {
            mySum.x = min(mySum.x, sdata[tid + 32].x);
            mySum.y = max(mySum.y, sdata[tid + 32].y);
            sdata[tid] = mySum;
        }
	 __syncthreads(); 
	
        if ((blockSize >=  32) && (tid < 16)) { 
            mySum.x = min(mySum.x, sdata[tid + 16].x);
            mySum.y = max(mySum.y, sdata[tid + 16].y);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=  16) && (tid <  8)) { 
            mySum.x = min(mySum.x, sdata[tid +  8].x);
            mySum.y = max(mySum.y, sdata[tid +  8].y);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=   8) && (tid <  4)) {
            mySum.x = min(mySum.x, sdata[tid +  4].x);
            mySum.y = max(mySum.y, sdata[tid +  4].y);
            sdata[tid] = mySum;
        }
	 __syncthreads(); 
	
       if ((blockSize >=   4) && (tid <  2)) { 
            mySum.x = min(mySum.x, sdata[tid +  2].x);
            mySum.y = max(mySum.y, sdata[tid +  2].y);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
	
        if ((blockSize >=   2) && ( tid <  1)) { 
            mySum.x = min(mySum.x, sdata[tid +  1].x);
            mySum.y = max(mySum.y, sdata[tid +  1].y);
            sdata[tid] = mySum;
        }
	__syncthreads(); 
#endif
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <class T>
void cuReduceFindMinMax(T *dst, T *src, unsigned numElements, unsigned numBlocks, unsigned numThreads)
{
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
	int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
	
	if (isPow2(numElements)) {
		switch (numThreads)
		{
		case 512:
			reduceFindMinMax_kernel<T,512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 256:
			reduceFindMinMax_kernel<T,256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 128:
			reduceFindMinMax_kernel<T,128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 64:
			reduceFindMinMax_kernel<T,64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 32:
			reduceFindMinMax_kernel<T,32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 16:
			reduceFindMinMax_kernel<T,16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  8:
			reduceFindMinMax_kernel<T, 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  4:
			reduceFindMinMax_kernel<T, 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  2:
			reduceFindMinMax_kernel<T, 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  1:
			reduceFindMinMax_kernel<T, 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		}
	}
	else {
		switch (numThreads)
		{
		case 512:
			reduceFindMinMax_kernel<T,512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 256:
			reduceFindMinMax_kernel<T,256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 128:
			reduceFindMinMax_kernel<T,128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 64:
			reduceFindMinMax_kernel<T,64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 32:
			reduceFindMinMax_kernel<T,32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 16:
			reduceFindMinMax_kernel<T,16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  8:
			reduceFindMinMax_kernel<T, 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  4:
			reduceFindMinMax_kernel<T, 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  2:
			reduceFindMinMax_kernel<T, 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  1:
			reduceFindMinMax_kernel<T, 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		}
	}
}

template void 
cuReduceFindMinMax<int2>(int2 *d_odata, int2 *d_idata, 
    unsigned numElements, unsigned numBlocks, unsigned numThreads);

template void 
cuReduceFindMinMax<float2>(float2 *d_odata, float2 *d_idata,
    unsigned numElements, unsigned numBlocks, unsigned numThreads);


template <class T, class T1>
void cuReduceFindMinMax(T *dst, T1 *src, unsigned numElements, unsigned numBlocks, unsigned numThreads)
{
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
	int smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);
	
	if (isPow2(numElements)) {
		switch (numThreads)
		{
		case 512:
			reduceFindMinMax_kernel1<T, T1, 512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 256:
			reduceFindMinMax_kernel1<T, T1, 256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 128:
			reduceFindMinMax_kernel1<T, T1, 128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 64:
			reduceFindMinMax_kernel1<T, T1, 64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 32:
			reduceFindMinMax_kernel1<T, T1, 32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 16:
			reduceFindMinMax_kernel1<T, T1, 16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  8:
			reduceFindMinMax_kernel1<T, T1,  8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  4:
			reduceFindMinMax_kernel1<T, T1,  4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  2:
			reduceFindMinMax_kernel1<T, T1,  2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  1:
			reduceFindMinMax_kernel1<T, T1,  1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		}
	}
	else {
		switch (numThreads)
		{
		case 512:
			reduceFindMinMax_kernel1<T, T1, 512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 256:
			reduceFindMinMax_kernel1<T, T1, 256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 128:
			reduceFindMinMax_kernel1<T, T1, 128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 64:
			reduceFindMinMax_kernel1<T, T1, 64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 32:
			reduceFindMinMax_kernel1<T, T1, 32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case 16:
			reduceFindMinMax_kernel1<T, T1, 16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  8:
			reduceFindMinMax_kernel1<T, T1,  8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  4:
			reduceFindMinMax_kernel1<T, T1,  4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  2:
			reduceFindMinMax_kernel1<T, T1,  2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		case  1:
			reduceFindMinMax_kernel1<T, T1,  1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numElements); break;
		}
	}
}

template void 
cuReduceFindMinMax<int2, int>(int2 *d_odata, int *d_idata, 
    unsigned numElements, unsigned numBlocks, unsigned numThreads);

template void 
cuReduceFindMinMax<float2, float>(float2 *d_odata, float *d_idata,
    unsigned numElements, unsigned numBlocks, unsigned numThreads);


