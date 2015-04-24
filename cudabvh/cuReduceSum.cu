#include "cuReduceSum_implement.h"

template <unsigned int blockSize, bool nIsPow2>
__global__ void reducePntMinX_kernel(float3 *g_idata, float *g_odata, uint n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    float myRange = 1e28f;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        myRange = min(myRange, g_idata[i].x);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            myRange = min(myRange, g_idata[i+blockSize].x);
        }
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = myRange;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { 
        if (tid < 256) { 
            myRange = min(myRange, sdata[tid + 256]);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            myRange = min(myRange, sdata[tid + 128]);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            myRange = min(myRange, sdata[tid +  64]); 
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }

    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float * smem = sdata;
        if (blockSize >=  64) {
            myRange = min(myRange, smem[tid + 32]);
            smem[tid] = myRange;
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
            myRange = min(myRange, smem[tid + 16]);
            smem[tid] = myRange;
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            myRange = min(myRange, smem[tid +  8]);
            smem[tid] = myRange;
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            myRange = min(myRange, smem[tid +  4]);
            smem[tid] = myRange;
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            myRange = min(myRange, smem[tid +  2]);
            smem[tid] = myRange;
           __syncthreads(); 
        }
        
        if (blockSize >=   2) { 
            smem[tid] = myRange = min(myRange, smem[tid +  1]);
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceBoxMinX_kernel(Aabb *g_idata, float *g_odata, uint n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    float myRange = 1e28f;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        myRange = min(myRange, g_idata[i].low.x);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            myRange = min(myRange, g_idata[i+blockSize].low.x);
        }
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = myRange;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { 
        if (tid < 256) { 
            myRange = min(myRange, sdata[tid + 256]);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            myRange = min(myRange, sdata[tid + 128]);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            myRange = min(myRange, sdata[tid +  64]); 
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }

    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float * smem = sdata;
        if (blockSize >=  64) {
            myRange = min(myRange, smem[tid + 32]);
            smem[tid] = myRange;
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
            myRange = min(myRange, smem[tid + 16]);
            smem[tid] = myRange;
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            myRange = min(myRange, smem[tid +  8]);
            smem[tid] = myRange;
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            myRange = min(myRange, smem[tid +  4]);
            smem[tid] = myRange;
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            myRange = min(myRange, smem[tid +  2]);
            smem[tid] = myRange;
           __syncthreads(); 
        }
        
        if (blockSize >=   2) { 
            smem[tid] = myRange = min(myRange, smem[tid +  1]);
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceFMin_kernel(float *g_idata, float *g_odata, uint n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    float myMin = 1e28f;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        myMin = min(myMin, g_idata[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            myMin = min(myMin, g_idata[i+blockSize]);  
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = myMin;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { 
        if (tid < 256) { 
            myMin = min(myMin, sdata[tid + 256]);
            sdata[tid] = myMin; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            myMin = min(myMin, sdata[tid + 128]); 
            sdata[tid] = myMin; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            myMin = min(myMin, sdata[tid +  64]); 
            sdata[tid] = myMin; 
        } 
        __syncthreads(); 
    }
    

    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float * smem = sdata;
        if (blockSize >=  64) {
            myMin = min(myMin, smem[tid + 32]);
            smem[tid] = myMin;
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
           myMin = min(myMin, smem[tid + 16]);
            smem[tid] = myMin;
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            myMin = min(myMin, smem[tid +  8]);
            smem[tid] = myMin;
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            myMin = min(myMin, smem[tid +  4]);
            smem[tid] = myMin;
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            myMin = min(myMin, smem[tid +  2]);
            smem[tid] = myMin;
            __syncthreads(); 
        }
        
        if (blockSize >=   2) { 
            myMin = min(myMin, smem[tid +  1]);
            smem[tid] = myMin;
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceFMax_kernel(float *g_idata, float *g_odata, uint n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    float myMax = -1e28f;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        myMax = max(myMax, g_idata[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            myMax = max(myMax, g_idata[i+blockSize]);  
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = myMax;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { 
        if (tid < 256) { 
            myMax = max(myMax, sdata[tid + 256]);
            sdata[tid] = myMax; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            myMax = max(myMax, sdata[tid + 128]); 
            sdata[tid] = myMax; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            myMax = max(myMax, sdata[tid +  64]); 
            sdata[tid] = myMax; 
        } 
        __syncthreads(); 
    }
    

    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float * smem = sdata;
        if (blockSize >=  64) {
            myMax = max(myMax, smem[tid + 32]);
            smem[tid] = myMax;
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
           myMax = max(myMax, smem[tid + 16]);
            smem[tid] = myMax;
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            myMax = max(myMax, smem[tid +  8]);
            smem[tid] = myMax;
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            myMax = max(myMax, smem[tid +  4]);
            smem[tid] = myMax;
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            myMax = max(myMax, smem[tid +  2]);
            smem[tid] = myMax;
            __syncthreads(); 
        }
        
        if (blockSize >=   2) { 
            myMax = max(myMax, smem[tid +  1]);
            smem[tid] = myMax;
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceFSum_kernel(float *g_idata, float *g_odata, uint n)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    float mySum = 0.f;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        mySum += g_idata[i];
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            mySum += g_idata[i+blockSize];  
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { 
        if (tid < 256) { 
            mySum += sdata[tid + 256];
            sdata[tid] = mySum; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            mySum += sdata[tid + 128]; 
            sdata[tid] = mySum; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            mySum += sdata[tid +  64]; 
            sdata[tid] = mySum; 
        } 
        __syncthreads(); 
    }
    

    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float * smem = sdata;
        if (blockSize >=  64) {
            mySum += smem[tid + 32];
            smem[tid] = mySum;
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
           mySum += smem[tid + 16];
            smem[tid] = mySum;
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            mySum += smem[tid +  8];
            smem[tid] = mySum;
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            mySum += smem[tid +  4];
            smem[tid] = mySum;
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            mySum += smem[tid +  2];
            smem[tid] = mySum;
            __syncthreads(); 
        }
        
        if (blockSize >=   2) { 
            mySum += smem[tid +  1];
            smem[tid] = mySum;
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

extern "C" {
    
void cuReduce_F_Sum(float *dst, float *src, 
    uint n, uint nBlocks, uint nThreads)
{
	dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
	uint smemSize = (nThreads <= 32) ? 64 * 4 : nThreads * 4;
	
	if (isPow2(n)) {
		switch (nThreads)
		{
		case 512:
			reduceFSum_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceFSum_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceFSum_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceFSum_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceFSum_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceFSum_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceFSum_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceFSum_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceFSum_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceFSum_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
	else {
		switch (nThreads)
		{
		case 512:
			reduceFSum_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceFSum_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceFSum_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceFSum_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceFSum_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceFSum_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceFSum_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceFSum_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceFSum_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceFSum_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
}

void cuReduce_F_Max(float * dst, float * src,
                    uint n, uint nBlocks, uint nThreads)
{
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
	uint smemSize = (nThreads <= 32) ? 64 * 4 : nThreads * 4;
	
	if (isPow2(n)) {
		switch (nThreads)
		{
		case 512:
			reduceFMax_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceFMax_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceFMax_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceFMax_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceFMax_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceFMax_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceFMax_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceFMax_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceFMax_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceFMax_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
	else {
		switch (nThreads)
		{
		case 512:
			reduceFMax_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceFMax_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceFMax_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceFMax_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceFMax_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceFMax_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceFMax_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceFMax_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceFMax_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceFMax_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
}


void cuReduce_F_Min(float * dst, float * src,
                    uint n, uint nBlocks, uint nThreads)
{
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
	uint smemSize = (nThreads <= 32) ? 64 * 4 : nThreads * 4;
	
	if (isPow2(n)) {
		switch (nThreads)
		{
		case 512:
			reduceFMin_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceFMin_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceFMin_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceFMin_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceFMin_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceFMin_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceFMin_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceFMin_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceFMin_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceFMin_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
	else {
		switch (nThreads)
		{
		case 512:
			reduceFMin_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceFMin_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceFMin_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceFMin_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceFMin_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceFMin_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceFMin_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceFMin_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceFMin_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceFMin_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
}

void cuReduce_Box_MinX(float * dst, Aabb * src,
                    uint n, uint nBlocks, uint nThreads)
{
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
	uint smemSize = (nThreads <= 32) ? 64 * 4 : nThreads * 4;
	
	if (isPow2(n)) {
		switch (nThreads)
		{
		case 512:
			reduceBoxMinX_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceBoxMinX_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceBoxMinX_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceBoxMinX_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceBoxMinX_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceBoxMinX_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceBoxMinX_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceBoxMinX_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceBoxMinX_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceBoxMinX_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
	else {
		switch (nThreads)
		{
		case 512:
			reduceBoxMinX_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceBoxMinX_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceBoxMinX_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceBoxMinX_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceBoxMinX_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceBoxMinX_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceBoxMinX_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceBoxMinX_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceBoxMinX_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceBoxMinX_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
}


void cuReduce_Pnt_MinX(float * dst, float3 * src,
                    uint n, uint nBlocks, uint nThreads)
{
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
	uint smemSize = (nThreads <= 32) ? 64 * 4 : nThreads * 4;
	
	if (isPow2(n)) {
		switch (nThreads)
		{
		case 512:
			reducePntMinX_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reducePntMinX_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reducePntMinX_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reducePntMinX_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reducePntMinX_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reducePntMinX_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reducePntMinX_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reducePntMinX_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reducePntMinX_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reducePntMinX_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
	else {
		switch (nThreads)
		{
		case 512:
			reducePntMinX_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reducePntMinX_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reducePntMinX_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reducePntMinX_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reducePntMinX_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reducePntMinX_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reducePntMinX_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reducePntMinX_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reducePntMinX_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reducePntMinX_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
}

}
