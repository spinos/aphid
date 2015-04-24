#include "cuReduceSum_implement.h"

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceBoxMin2F4_kernel(float4 *g_idata, float4 *g_odata, uint n)
{
    extern __shared__ float4 sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    float4 myRange = make_float4(1e28f, 1e28f, 1e28f, 1.f);

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        myRange.x = min(myRange.x, g_idata[i].x);
        myRange.y = min(myRange.y, g_idata[i].y);
        myRange.z = min(myRange.z, g_idata[i].z);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            myRange.x = min(myRange.x, g_idata[i+blockSize].x);
            myRange.y = min(myRange.y, g_idata[i+blockSize].y);
            myRange.z = min(myRange.z, g_idata[i+blockSize].z);
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
            myRange.y = min(myRange.y, sdata[tid + 256].y);
            myRange.z = min(myRange.y, sdata[tid + 256].z);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            myRange.x = min(myRange.x, sdata[tid + 128].x);
            myRange.y = min(myRange.y, sdata[tid + 128].y);
            myRange.z = min(myRange.z, sdata[tid + 128].z);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            myRange.x = min(myRange.x, sdata[tid +  64].x); 
            myRange.y = min(myRange.y, sdata[tid + 64].y);
            myRange.z = min(myRange.z, sdata[tid + 64].z);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    

    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float4 * smem = sdata;
        if (blockSize >=  64) {
            myRange.x = min(myRange.x, smem[tid + 32].x);
            myRange.y = min(myRange.y, smem[tid + 32].y);
            myRange.z = min(myRange.y, smem[tid + 32].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
            myRange.x = min(myRange.x, smem[tid + 16].x);
            myRange.y = min(myRange.y, smem[tid + 16].y);
            myRange.z = min(myRange.z, smem[tid + 16].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            myRange.x = min(myRange.x, smem[tid +  8].x);
            myRange.y = min(myRange.y, smem[tid +  8].y);
            myRange.z = min(myRange.z, smem[tid +  8].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            myRange.x = min(myRange.x, smem[tid +  4].x);
            myRange.y = min(myRange.y, smem[tid +  4].y);
            myRange.z = min(myRange.z, smem[tid +  4].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            myRange.x = min(myRange.x, smem[tid +  2].x);
            myRange.y = min(myRange.y, smem[tid +  2].y);
            myRange.z = min(myRange.z, smem[tid +  2].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        
        if (blockSize >=   2) { 
            myRange.x = min(myRange.x, smem[tid +  1].x);
            myRange.y = min(myRange.y, smem[tid +  1].y);
            myRange.z = min(myRange.y, smem[tid +  1].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceBoxMin1F4_kernel(Aabb *g_idata, float4 *g_odata, uint n)
{
    extern __shared__ float4 sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    float4 myRange = make_float4(1e28f, 1e28f, 1e28f, 1.f);

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        myRange.x = min(myRange.x, g_idata[i].low.x);
        myRange.y = min(myRange.y, g_idata[i].low.y);
        myRange.z = min(myRange.z, g_idata[i].low.z);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            myRange.x = min(myRange.x, g_idata[i+blockSize].low.x);
            myRange.y = min(myRange.y, g_idata[i+blockSize].low.y);
            myRange.z = min(myRange.z, g_idata[i+blockSize].low.z);
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
            myRange.y = min(myRange.y, sdata[tid + 256].y);
            myRange.z = min(myRange.y, sdata[tid + 256].z);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            myRange.x = min(myRange.x, sdata[tid + 128].x);
            myRange.y = min(myRange.y, sdata[tid + 128].y);
            myRange.z = min(myRange.z, sdata[tid + 128].z);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            myRange.x = min(myRange.x, sdata[tid +  64].x); 
            myRange.y = min(myRange.y, sdata[tid + 64].y);
            myRange.z = min(myRange.z, sdata[tid + 64].z);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    

    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float4 * smem = sdata;
        if (blockSize >=  64) {
            myRange.x = min(myRange.x, smem[tid + 32].x);
            myRange.y = min(myRange.y, smem[tid + 32].y);
            myRange.z = min(myRange.y, smem[tid + 32].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
            myRange.x = min(myRange.x, smem[tid + 16].x);
            myRange.y = min(myRange.y, smem[tid + 16].y);
            myRange.z = min(myRange.z, smem[tid + 16].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            myRange.x = min(myRange.x, smem[tid +  8].x);
            myRange.y = min(myRange.y, smem[tid +  8].y);
            myRange.z = min(myRange.z, smem[tid +  8].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            myRange.x = min(myRange.x, smem[tid +  4].x);
            myRange.y = min(myRange.y, smem[tid +  4].y);
            myRange.z = min(myRange.z, smem[tid +  4].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            myRange.x = min(myRange.x, smem[tid +  2].x);
            myRange.y = min(myRange.y, smem[tid +  2].y);
            myRange.z = min(myRange.z, smem[tid +  2].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        
        if (blockSize >=   2) { 
            myRange.x = min(myRange.x, smem[tid +  1].x);
            myRange.y = min(myRange.y, smem[tid +  1].y);
            myRange.z = min(myRange.y, smem[tid +  1].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceBoxMax2F4_kernel(float4 *g_idata, float4 *g_odata, uint n)
{
    extern __shared__ float4 sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    float4 myRange = make_float4(-1e28f, -1e28f, -1e28f, 1.f);

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        myRange.x = max(myRange.x, g_idata[i].x);
        myRange.y = max(myRange.y, g_idata[i].y);
        myRange.z = max(myRange.z, g_idata[i].z);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            myRange.x = max(myRange.x, g_idata[i+blockSize].x);
            myRange.y = max(myRange.y, g_idata[i+blockSize].y);
            myRange.z = max(myRange.z, g_idata[i+blockSize].z);
        }
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = myRange;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { 
        if (tid < 256) { 
            myRange.x = max(myRange.x, sdata[tid + 256].x);
            myRange.y = max(myRange.y, sdata[tid + 256].y);
            myRange.z = max(myRange.y, sdata[tid + 256].z);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            myRange.x = max(myRange.x, sdata[tid + 128].x);
            myRange.y = max(myRange.y, sdata[tid + 128].y);
            myRange.z = max(myRange.z, sdata[tid + 128].z);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            myRange.x = max(myRange.x, sdata[tid +  64].x); 
            myRange.y = max(myRange.y, sdata[tid + 64].y);
            myRange.z = max(myRange.z, sdata[tid + 64].z);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    

    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float4 * smem = sdata;
        if (blockSize >=  64) {
            myRange.x = max(myRange.x, smem[tid + 32].x);
            myRange.y = max(myRange.y, smem[tid + 32].y);
            myRange.z = max(myRange.y, smem[tid + 32].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
            myRange.x = max(myRange.x, smem[tid + 16].x);
            myRange.y = max(myRange.y, smem[tid + 16].y);
            myRange.z = max(myRange.z, smem[tid + 16].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            myRange.x = max(myRange.x, smem[tid +  8].x);
            myRange.y = max(myRange.y, smem[tid +  8].y);
            myRange.z = max(myRange.z, smem[tid +  8].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            myRange.x = max(myRange.x, smem[tid +  4].x);
            myRange.y = max(myRange.y, smem[tid +  4].y);
            myRange.z = max(myRange.z, smem[tid +  4].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            myRange.x = max(myRange.x, smem[tid +  2].x);
            myRange.y = max(myRange.y, smem[tid +  2].y);
            myRange.z = max(myRange.z, smem[tid +  2].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        
        if (blockSize >=   2) { 
            myRange.x = max(myRange.x, smem[tid +  1].x);
            myRange.y = max(myRange.y, smem[tid +  1].y);
            myRange.z = max(myRange.y, smem[tid +  1].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceBoxMax1F4_kernel(Aabb *g_idata, float4 *g_odata, uint n)
{
    extern __shared__ float4 sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    float4 myRange = make_float4(-1e28f, -1e28f, -1e28f, 1.f);

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        myRange.x = max(myRange.x, g_idata[i].high.x);
        myRange.y = max(myRange.y, g_idata[i].high.y);
        myRange.z = max(myRange.z, g_idata[i].high.z);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) {
            myRange.x = max(myRange.x, g_idata[i+blockSize].high.x);
            myRange.y = max(myRange.y, g_idata[i+blockSize].high.y);
            myRange.z = max(myRange.z, g_idata[i+blockSize].high.z);
        }
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = myRange;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { 
        if (tid < 256) { 
            myRange.x = max(myRange.x, sdata[tid + 256].x);
            myRange.y = max(myRange.y, sdata[tid + 256].y);
            myRange.z = max(myRange.y, sdata[tid + 256].z);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            myRange.x = max(myRange.x, sdata[tid + 128].x);
            myRange.y = max(myRange.y, sdata[tid + 128].y);
            myRange.z = max(myRange.z, sdata[tid + 128].z);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            myRange.x = max(myRange.x, sdata[tid + 64].x); 
            myRange.y = max(myRange.y, sdata[tid + 64].y);
            myRange.z = max(myRange.z, sdata[tid + 64].z);
            sdata[tid] = myRange; 
        } 
        __syncthreads(); 
    }
    

    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile float4 * smem = sdata;
        if (blockSize >=  64) {
            myRange.x = max(myRange.x, smem[tid + 32].x);
            myRange.y = max(myRange.y, smem[tid + 32].y);
            myRange.z = max(myRange.y, smem[tid + 32].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
            myRange.x = max(myRange.x, smem[tid + 16].x);
            myRange.y = max(myRange.y, smem[tid + 16].y);
            myRange.z = max(myRange.z, smem[tid + 16].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            myRange.x = max(myRange.x, smem[tid +  8].x);
            myRange.y = max(myRange.y, smem[tid +  8].y);
            myRange.z = max(myRange.z, smem[tid +  8].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            myRange.x = max(myRange.x, smem[tid +  4].x);
            myRange.y = max(myRange.y, smem[tid +  4].y);
            myRange.z = max(myRange.z, smem[tid +  4].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            myRange.x = max(myRange.x, smem[tid +  2].x);
            myRange.y = max(myRange.y, smem[tid +  2].y);
            myRange.z = max(myRange.z, smem[tid +  2].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
        
        if (blockSize >=   2) { 
            myRange.x = max(myRange.x, smem[tid +  1].x);
            myRange.y = max(myRange.y, smem[tid +  1].y);
            myRange.z = max(myRange.y, smem[tid +  1].z);
            smem[tid].x = myRange.x;
            smem[tid].y = myRange.y;
            smem[tid].z = myRange.z;
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

extern "C" {
    
void cuReduce_Box_Min1_Flt4(float4 * dst, Aabb * src,
                    uint n, uint nBlocks, uint nThreads)
{
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
	uint smemSize = (nThreads <= 32) ? 64 * 16 : nThreads * 16;
	
	if (isPow2(n)) {
		switch (nThreads)
		{
		case 512:
			reduceBoxMin1F4_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceBoxMin1F4_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceBoxMin1F4_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceBoxMin1F4_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceBoxMin1F4_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceBoxMin1F4_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceBoxMin1F4_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceBoxMin1F4_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceBoxMin1F4_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceBoxMin1F4_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
	else {
		switch (nThreads)
		{
		case 512:
			reduceBoxMin1F4_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceBoxMin1F4_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceBoxMin1F4_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceBoxMin1F4_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceBoxMin1F4_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceBoxMin1F4_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceBoxMin1F4_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceBoxMin1F4_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceBoxMin1F4_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceBoxMin1F4_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
}

void cuReduce_Box_Min2_Flt4(float4 * dst, float4 * src,
                    uint n, uint nBlocks, uint nThreads)
{
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
	uint smemSize = (nThreads <= 32) ? 64 * 16 : nThreads * 16;
	
	if (isPow2(n)) {
		switch (nThreads)
		{
		case 512:
			reduceBoxMin2F4_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceBoxMin2F4_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceBoxMin2F4_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceBoxMin2F4_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceBoxMin2F4_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceBoxMin2F4_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceBoxMin2F4_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceBoxMin2F4_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceBoxMin2F4_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceBoxMin2F4_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
	else {
		switch (nThreads)
		{
		case 512:
			reduceBoxMin2F4_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceBoxMin2F4_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceBoxMin2F4_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceBoxMin2F4_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceBoxMin2F4_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceBoxMin2F4_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceBoxMin2F4_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceBoxMin2F4_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceBoxMin2F4_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceBoxMin2F4_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
}

void cuReduce_Box_Max1_Flt4(float4 * dst, Aabb * src,
                    uint n, uint nBlocks, uint nThreads)
{
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
	uint smemSize = (nThreads <= 32) ? 64 * 16 : nThreads * 16;
	
	if (isPow2(n)) {
		switch (nThreads)
		{
		case 512:
			reduceBoxMax1F4_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceBoxMax1F4_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceBoxMax1F4_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceBoxMax1F4_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceBoxMax1F4_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceBoxMax1F4_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceBoxMax1F4_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceBoxMax1F4_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceBoxMax1F4_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceBoxMax1F4_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
	else {
		switch (nThreads)
		{
		case 512:
			reduceBoxMax1F4_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceBoxMax1F4_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceBoxMax1F4_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceBoxMax1F4_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceBoxMax1F4_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceBoxMax1F4_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceBoxMax1F4_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceBoxMax1F4_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceBoxMax1F4_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceBoxMax1F4_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
}

void cuReduce_Box_Max2_Flt4(float4 * dst, float4 * src,
                    uint n, uint nBlocks, uint nThreads)
{
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
	uint smemSize = (nThreads <= 32) ? 64 * 16 : nThreads * 16;
	
	if (isPow2(n)) {
		switch (nThreads)
		{
		case 512:
			reduceBoxMax2F4_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceBoxMax2F4_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceBoxMax2F4_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceBoxMax2F4_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceBoxMax2F4_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceBoxMax2F4_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceBoxMax2F4_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceBoxMax2F4_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceBoxMax2F4_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceBoxMax2F4_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
	else {
		switch (nThreads)
		{
		case 512:
			reduceBoxMax2F4_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 256:
			reduceBoxMax2F4_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 128:
			reduceBoxMax2F4_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 64:
			reduceBoxMax2F4_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 32:
			reduceBoxMax2F4_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case 16:
			reduceBoxMax2F4_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  8:
			reduceBoxMax2F4_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  4:
			reduceBoxMax2F4_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  2:
			reduceBoxMax2F4_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		case  1:
			reduceBoxMax2F4_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, n); break;
		}
	}
}

}
