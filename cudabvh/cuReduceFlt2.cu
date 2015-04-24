#include "cuReduceSum_implement.h"

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceFMinMax2_kernel(float2 *g_idata, float2 *g_odata, uint n)
{
    extern __shared__ float2 sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    float2 myRange = make_float2(1e28f, -1e28f);

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        myRange.x = min(myRange.x, g_idata[i].x);
        myRange.y = max(myRange.y, g_idata[i].y);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            myRange.x = min(myRange.x, g_idata[i+blockSize].x);
            myRange.y = max(myRange.y, g_idata[i+blockSize].y);
        }
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = myRange;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { 
        if (tid < 256) { 
            myRange.x = min(myRange.x, sdata[tid + 256].x);
            myRange.y = max(myRange.y, sdata[tid + 256].y);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            myRange.x = min(myRange.x, sdata[tid + 128].x);
            myRange.y = max(myRange.y, sdata[tid + 128].y);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            myRange.x = min(myRange.x, sdata[tid +  64].x); 
            myRange.y = max(myRange.y, sdata[tid + 64].y);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    

    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float2 * smem = sdata;
        if (blockSize >=  64) {
            myRange.x = min(myRange.x, smem[tid + 32].x);
            myRange.y = max(myRange.y, smem[tid + 32].y);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
            myRange.x = min(myRange.x, smem[tid + 16].x);
            myRange.y = max(myRange.y, smem[tid + 16].y);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            myRange.x = min(myRange.x, smem[tid +  8].x);
            myRange.y = max(myRange.y, smem[tid +  8].y);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            myRange.x = min(myRange.x, smem[tid +  4].x);
            myRange.y = max(myRange.y, smem[tid +  4].y);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            myRange.x = min(myRange.x, smem[tid +  2].x);
            myRange.y = max(myRange.y, smem[tid +  2].y);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        
        if (blockSize >=   2) { 
            myRange.x = min(myRange.x, smem[tid +  1].x);
            myRange.y = max(myRange.y, smem[tid +  1].y);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceFMinMax1_kernel(float *g_idata, float2 *g_odata, uint n)
{
    extern __shared__ float2 sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    float2 myRange = make_float2(1e28f, -1e28f);

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        myRange.x = min(myRange.x, g_idata[i]);
        myRange.y = max(myRange.y, g_idata[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            myRange.x = min(myRange.x, g_idata[i+blockSize]);
            myRange.y = max(myRange.y, g_idata[i+blockSize]);
        }
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = myRange;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { 
        if (tid < 256) { 
            myRange.x = min(myRange.x, sdata[tid + 256].x);
            myRange.y = max(myRange.y, sdata[tid + 256].y);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            myRange.x = min(myRange.x, sdata[tid + 128].x);
            myRange.y = max(myRange.y, sdata[tid + 128].y);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            myRange.x = min(myRange.x, sdata[tid +  64].x); 
            myRange.y = max(myRange.y, sdata[tid + 64].y);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    

    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float2 * smem = sdata;
        if (blockSize >=  64) {
            myRange.x = min(myRange.x, smem[tid + 32].x);
            myRange.y = max(myRange.y, smem[tid + 32].y);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
            myRange.x = min(myRange.x, smem[tid + 16].x);
            myRange.y = max(myRange.y, smem[tid + 16].y);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            myRange.x = min(myRange.x, smem[tid +  8].x);
            myRange.y = max(myRange.y, smem[tid +  8].y);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            myRange.x = min(myRange.x, smem[tid +  4].x);
            myRange.y = max(myRange.y, smem[tid +  4].y);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            myRange.x = min(myRange.x, smem[tid +  2].x);
            myRange.y = max(myRange.y, smem[tid +  2].y);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        
        if (blockSize >=   2) { 
            myRange.x = min(myRange.x, smem[tid +  1].x);
            myRange.y = max(myRange.y, smem[tid +  1].y);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

extern "C" {
    
void cuReduce_F_MinMax1(float2 * dst, float * src,
                    uint n, uint nBlocks, uint nThreads)
{
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
	uint smemSize = (nThreads <= 32) ? 64 * 8 : nThreads * 8;
	
	if (isPow2(n)) {
		switch (nThreads)
		{
		case 512:
			reduceFMinMax1_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceFMinMax1_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceFMinMax1_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceFMinMax1_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceFMinMax1_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceFMinMax1_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceFMinMax1_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceFMinMax1_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceFMinMax1_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceFMinMax1_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
	else {
		switch (nThreads)
		{
		case 512:
			reduceFMinMax1_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceFMinMax1_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceFMinMax1_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceFMinMax1_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceFMinMax1_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceFMinMax1_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceFMinMax1_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceFMinMax1_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceFMinMax1_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceFMinMax1_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
}

void cuReduce_F_MinMax2(float2 * dst, float2 * src,
                    uint n, uint nBlocks, uint nThreads)
{
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
	uint smemSize = (nThreads <= 32) ? 64 * 8 : nThreads * 8;
	
	if (isPow2(n)) {
		switch (nThreads)
		{
		case 512:
			reduceFMinMax2_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceFMinMax2_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceFMinMax2_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceFMinMax2_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceFMinMax2_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceFMinMax2_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceFMinMax2_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceFMinMax2_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceFMinMax2_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceFMinMax2_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
	else {
		switch (nThreads)
		{
		case 512:
			reduceFMinMax2_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceFMinMax2_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceFMinMax2_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceFMinMax2_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceFMinMax2_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceFMinMax2_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceFMinMax2_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceFMinMax2_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceFMinMax2_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceFMinMax2_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
}

}
