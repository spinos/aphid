#include "createBvh_implement.h"

inline __device__ void resetAabb(Aabb & dst)
{
    dst.low = make_float3(10e9, 10e9, 10e9);
    dst.high = make_float3(-10e9, -10e9, -10e9);
}

inline __device__ void expandAabb(Aabb & dst, float4 p)
{
    if(p.x < dst.low.x) dst.low.x = p.x;
    if(p.y < dst.low.y) dst.low.y = p.y;
    if(p.z < dst.low.z) dst.low.z = p.z;
    if(p.x > dst.high.x) dst.high.x = p.x;
    if(p.y > dst.high.y) dst.high.y = p.y;
    if(p.z > dst.high.z) dst.high.z = p.z;
}

inline __device__ void expandAabb(Aabb & dst, Aabb src)
{
    if(src.low.x < dst.low.x) dst.low.x = src.low.x;
    if(src.low.y < dst.low.y) dst.low.y = src.low.y;
    if(src.low.z < dst.low.z) dst.low.z = src.low.z;
    if(src.high.x > dst.high.x) dst.high.x = src.high.x;
    if(src.high.y > dst.high.y) dst.high.y = src.high.y;
    if(src.high.z > dst.high.z) dst.high.z = src.high.z;
}

inline __device__ void expandAabb(Aabb & dst, volatile Aabb * src)
{
    if(src->low.x < dst.low.x) dst.low.x = src->low.x;
    if(src->low.y < dst.low.y) dst.low.y = src->low.y;
    if(src->low.z < dst.low.z) dst.low.z = src->low.z;
    if(src->high.x > dst.high.x) dst.high.x = src->high.x;
    if(src->high.y > dst.high.y) dst.high.y = src->high.y;
    if(src->high.z > dst.high.z) dst.high.z = src->high.z;
}

inline __device__ void copyVola(volatile Aabb * dst, const Aabb & src)
{
    dst->low.x = src.low.x;
    dst->low.y = src.low.y;
    dst->low.z = src.low.z;
    dst->high.x = src.high.x;
    dst->high.y = src.high.y;
    dst->high.z = src.high.z;
}

__global__ void calculateAabbs_kernel(Aabb *dst, float4 * cvs, EdgeContact * edges, unsigned maxEdgeInd, unsigned maxVertInd)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

	if(idx >= maxEdgeInd) return;
	
	EdgeContact e = edges[idx];
	unsigned v0 = e.v[0];
	unsigned v1 = e.v[1];
	unsigned v2 = e.v[2];
	unsigned v3 = e.v[3];
	
	Aabb res;
	resetAabb(res);
	if(v0 < maxVertInd) expandAabb(res, cvs[v0]);
	if(v1 < maxVertInd) expandAabb(res, cvs[v1]);
	if(v2 < maxVertInd) expandAabb(res, cvs[v2]);
	if(v3 < maxVertInd) expandAabb(res, cvs[v3]);
	
	dst[idx] = res;
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceAabb_kernel(Aabb *g_idata, Aabb *g_odata, unsigned int n)
{
    extern __shared__ Aabb sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    
    Aabb mySum; resetAabb(mySum);

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {         
        expandAabb(mySum, g_idata[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n) 
            expandAabb(mySum, g_idata[i+blockSize]);  
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { 
        if (tid < 256) { 
            expandAabb(mySum, sdata[tid + 256]);
            sdata[tid] = mySum; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) { 
        if (tid < 128) { 
            expandAabb(mySum, sdata[tid + 128]); 
            sdata[tid] = mySum; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) { 
        if (tid <  64) { 
            expandAabb(mySum, sdata[tid +  64]); 
            sdata[tid] = mySum; 
        } 
        __syncthreads(); 
    }
    
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile Aabb * smem = sdata;
        if (blockSize >=  64) {
            expandAabb(mySum, &smem[tid + 32]);
            copyVola(&smem[tid], mySum);
            __syncthreads(); 
        }
        if (blockSize >=  32) { 
            expandAabb(mySum, &smem[tid + 16]);
            copyVola(&smem[tid], mySum);
            __syncthreads(); 
        }
        if (blockSize >=  16) { 
            expandAabb(mySum, &smem[tid +  8]);
            copyVola(&smem[tid], mySum);
            __syncthreads(); 
        }
        if (blockSize >=   8) { 
            expandAabb(mySum, &smem[tid +  4]);
            copyVola(&smem[tid], mySum);
            __syncthreads(); 
        }
        if (blockSize >=   4) { 
            expandAabb(mySum, &smem[tid +  2]);
            copyVola(&smem[tid], mySum);
            __syncthreads(); 
        }
        if (blockSize >=   2) { 
            expandAabb(mySum, &smem[tid +  1]);
            copyVola(&smem[tid], mySum);
            __syncthreads(); 
        }
    }
    
    // write result for this block to global mem 
    if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}

extern "C" void bvhCalculateAabbs(Aabb *dst, float4 * cvs, EdgeContact * edges, unsigned numEdges, unsigned numVertices)
{
    dim3 block(512, 1, 1);
    unsigned nblk = iDivUp(numEdges, 512);
    
    dim3 grid(nblk, 1, 1);
    calculateAabbs_kernel<<< grid, block >>>(dst, cvs, edges, numEdges, numVertices);
}

extern "C" void bvhReduceAabb(Aabb *dst, Aabb *src, unsigned numAabbs, unsigned numBlocks, unsigned numThreads)
{
	dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);
	int smemSize = (numThreads <= 2) ? 2 * numThreads * sizeof(Aabb) : numThreads * sizeof(Aabb);
	
	if (isPow2(numAabbs)) {
		switch (numThreads)
		{
		case 512:
			reduceAabb_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case 256:
			reduceAabb_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case 128:
			reduceAabb_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case 64:
			reduceAabb_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case 32:
			reduceAabb_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case 16:
			reduceAabb_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case  8:
			reduceAabb_kernel< 8, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case  4:
			reduceAabb_kernel< 4, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case  2:
			reduceAabb_kernel< 2, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case  1:
			reduceAabb_kernel< 1, true><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		}
	}
	else {
		switch (numThreads)
		{
		case 512:
			reduceAabb_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case 256:
			reduceAabb_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case 128:
			reduceAabb_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case 64:
			reduceAabb_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case 32:
			reduceAabb_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case 16:
			reduceAabb_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case  8:
			reduceAabb_kernel< 8, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case  4:
			reduceAabb_kernel< 4, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case  2:
			reduceAabb_kernel< 2, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		case  1:
			reduceAabb_kernel< 1, false><<< dimGrid, dimBlock, smemSize >>>(src, dst, numAabbs); break;
		}
	}
}

extern "C" void getReduceBlockThread(uint & blocks, uint & threads, uint n)
{
	uint maxThreads = 512;
	uint maxBlocks = 64;
	threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);
	if(blocks > maxBlocks) blocks = maxBlocks;
}
