#include "matrix_math.cuh"
#include "ray_intersection.cuh"
#include "cuSMem.cuh"
#include "cuReduceInBlock.cuh"
#include "ray_triangle.cuh"
#include <radixsort_implement.h>

#define ADE_RAYTRAVERSE_MAX_STACK_SIZE 64

__constant__ mat44 c_modelViewMatrix;  // inverse view matrix

inline __device__ int outOfStack(int stackSize)
{return (stackSize < 1 || stackSize > ADE_RAYTRAVERSE_MAX_STACK_SIZE); }

inline __device__ int isInternalNode(int2 child)
{ return (child.x>>31) != 0; }

inline __device__ int iLeafNode(int2 child)
{ return (child.x>>31) == 0; }

__device__ uint tId()
{ return blockDim.x * threadIdx.y + threadIdx.x; }

__device__ int intersectLeafTriangles(float & rayLength,
                       const Ray & eyeRay,
                       int2 range,
                       KeyValuePair * elementHash,
                       int4 * elementVertices,
                       float3 * elementPoints)
{
    int stat = 0;
    float u, v, t;
    int frontFacing;
    uint iElement;
    int4 triVert;
    int i=range.x;
    for(;i<=range.y;i++) {
        iElement = elementHash[i].value;
        triVert = elementVertices[iElement];
        if(ray_triangle_MollerTrumbore(eyeRay,
            elementPoints[triVert.x],
            elementPoints[triVert.y],
            elementPoints[triVert.z],
            u, v, t, frontFacing)) {
            if(t<rayLength) {
                rayLength = t;
                stat = 1;
            }
        }
    }
    return stat;   
}

__global__ void resetImage_kernel(float4 * pix, 
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	pix[ind] = make_float4(0.f, 0.f, 0.f, 1e20f);
}

template<int NumThreads>
__global__ void renderImageOrthographic_kernel(float4 * pix,
                uint imageW,
                uint imageH,
                float fovWidth,
                float aspectRatio,
                int2 * internalNodeChildIndices,
				Aabb * internalNodeAabbs,
				KeyValuePair * elementHash,
				int4 * elementVertices,
				float3 * elementPoints)
{
    int *sdata = SharedMemory<int>();
    
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= imageW) || (y >= imageH)) return;
    
    float u = (x / (float) imageW) - .5f;
    float v = (y / (float) imageH) - .5f;
    
    Ray eyeRay;
    eyeRay.o = make_float4(u * fovWidth, v * fovWidth/aspectRatio, 0.f, 1.f);
    eyeRay.d = make_float4(0.f, 0.f, -1.f, 0.f);
    
    eyeRay.o = transform(c_modelViewMatrix, eyeRay.o);
    eyeRay.d = transform(c_modelViewMatrix, eyeRay.d);
    normalize(eyeRay.d);
  
    float4 outRgbz = pix[y * imageW + x];
    float rayLength = outRgbz.w;
/* 
 *  smem layout in ints
 *  n as num threads
 *  m as max stack size
 *
 *  0   -> 1      stackSize
 *  1   -> m      stack
 *  m+1 -> m+1+n  branching
 *  m+1+n+1 -> m+1+n+1+n  visiting
 *
 *  branching is first child to visit
 *  -1 left 1 right 0 neither
 *  if sum(branching) < 1 left first else right first
 *  visiting is n child to visit
 *  3 both 2 left 1 right 0 neither
 *  if max(visiting) == 0 pop stack
 *  if max(visiting) == 1 override top of stack by right
 *  if max(visiting) >= 2 override top of stack by second, then push first to stack
 */
    int & sstackSize = sdata[0];
    int * sstack = &sdata[1];
    int * sbranch = &sdata[ADE_RAYTRAVERSE_MAX_STACK_SIZE+1];
    int * svisit = &sdata[ADE_RAYTRAVERSE_MAX_STACK_SIZE+1+NumThreads+1];
    const uint tid = tId();
    if(tid<1) {
        sstack[0] = 0x80000000;
        sstackSize = 1;
    }
    __syncthreads();

    int isLeaf;
    int iNode;
    int2 child;
    int2 pushChild;
    Aabb leftBox, rightBox;
    float lambda1, lambda2;
    float mu1, mu2;
    int b1, b2;
    for(;;) {
        iNode = sstack[ sstackSize - 1 ];
		
		iNode = getIndexWithInternalNodeMarkerRemoved(iNode);
        child = internalNodeChildIndices[iNode];
        isLeaf = iLeafNode(child);
		
        if(isLeaf) {
            leftBox = internalNodeAabbs[iNode];
            b1 = ray_box(lambda1, lambda2,
                    eyeRay,
                    rayLength,
                    leftBox);
            if(b1) {
// intersect triangles in leaf  
                if(intersectLeafTriangles(rayLength,
                                eyeRay, 
                                child,
                                elementHash,
                                elementVertices,
                                elementPoints)) {
                    outRgbz.x = rayLength/300.f;
                    outRgbz.y = rayLength/300.f;
                    outRgbz.z = rayLength/300.f;
                    outRgbz.w = rayLength;
                }
            }
            
            if(tid<1) {
// take out top of stack
                sstackSize--;
            }
            __syncthreads();
            
            if(sstackSize<1) break;
        }
        else {
            leftBox = internalNodeAabbs[getIndexWithInternalNodeMarkerRemoved(child.x)];
            b1 = ray_box(lambda1, lambda2,
                    eyeRay,
                    rayLength,
                    leftBox);
            
            rightBox = internalNodeAabbs[getIndexWithInternalNodeMarkerRemoved(child.y)];
            b2 = ray_box(mu1, mu2,
                    eyeRay,
                    rayLength,
                    rightBox);
            
            svisit[tid] = 2 * b1 + b2;
            if(svisit[tid]==3) { 
// visit both children
                if(mu1 < lambda1) {
// vist right child first
                    sbranch[tid] = 1;
                }
                else {
// vist left child first
                    sbranch[tid] = -1;
                }
            }
            else if(svisit[tid]==2) { 
// visit left child
                sbranch[tid] = -1;
            }
            else if(svisit[tid]==1) { 
// visit right child
                sbranch[tid] = 1;
            }
            else { 
// visit no child
                sbranch[tid] = 0;
            }
            __syncthreads();
            
// branching decision
            reduceSumInBlock<NumThreads, int>(tid, sbranch);
            reduceMaxInBlock<NumThreads, int>(tid, svisit);
            if(tid<1) {
                if(svisit[tid] == 0) {
// visit no child, take out top of stack
                    sstackSize--;
                }
                else if(svisit[tid] == 1) {
// visit right child
                    sstack[ sstackSize - 1 ] = child.y; 
                }
                else {
// visit both children
                    if(sbranch[tid]<1) {
                        pushChild = child;
                    }
                    else {
                        pushChild.x = child.y;
                        pushChild.y = child.x;
                    }
                    
                    sstack[ sstackSize - 1 ] = pushChild.y;
                    if(sstackSize < ADE_RAYTRAVERSE_MAX_STACK_SIZE) { 
                            sstack[ sstackSize ] = pushChild.x;
                            sstackSize++;
                    }
                }
            }
            
            __syncthreads();
            
            if(sstackSize<1) break;
        }
    }
    pix[y * imageW + x] = outRgbz;
}

