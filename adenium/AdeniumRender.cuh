#include "matrix_math.cuh"
#include "ray_intersection.cuh"
#include "cuSMem.cuh"
#include "cuReduceInBlock.cuh"
#include "ray_triangle.cuh"
#include <radixsort_implement.h>

#define ADE_RAYTRAVERSE_MAX_STACK_SIZE 64
#define ADE_RAYTRAVERSE_TRIANGLE_CACHE_SIZE 128

__constant__ mat44 c_modelViewMatrix;  // inverse view matrix
__constant__ int2 c_imageSize;
__constant__ float2 c_cameraProp; // (fieldOfView, aspectRatio)

inline __device__ int outOfStack(int stackSize)
{return (stackSize < 1 || stackSize > ADE_RAYTRAVERSE_MAX_STACK_SIZE); }

inline __device__ int isInternalNode(int2 child)
{ return (child.x>>31) != 0; }

inline __device__ int iLeafNode(int2 child)
{ return (child.x>>31) == 0; }

__device__ uint tId()
{ return blockDim.x * threadIdx.y + threadIdx.x; }

template<int NumThreads>
__device__ void putLeafTrianglesInSmem(float3 * points,
                                        uint tid,
                                        int n,
                                        int2 range,
                                        KeyValuePair * elementHash,
                                        int4 * elementVertices,
                                        float3 * elementPoints)
{
    uint iElement; 
    int4 triVert;
    uint loc = tid;
    
    if(loc < n) {
        iElement = elementHash[range.x + loc].value;
        triVert= elementVertices[iElement];
        points[loc*3] = elementPoints[triVert.x];
        points[loc*3+1] = elementPoints[triVert.y];
        points[loc*3+2] = elementPoints[triVert.z];
    }
    
    if(n>NumThreads) {
        loc += NumThreads;
        if(loc < n) {
            iElement = elementHash[range.x + loc].value;
            triVert= elementVertices[iElement];
            points[loc*3] = elementPoints[triVert.x];
            points[loc*3+1] = elementPoints[triVert.y];
            points[loc*3+2] = elementPoints[triVert.z];
        }
    }
}

__device__ int intersectLeafTrianglesS(float & rayLength,
                                int & triId,
                                const Ray & eyeRay,
                                int cacheLength,
                                float3 * trianglePoints)
{
    int stat = 0;
    float u, v, t;
    int frontFacing;
    int i=0;
    for(;i<cacheLength;i++) {
        if(ray_triangle_MollerTrumbore(eyeRay,
            trianglePoints[i*3],
            trianglePoints[i*3+1],
            trianglePoints[i*3+2],
            u, v, t, frontFacing)) {
            if(t<rayLength) {
                rayLength = t;
                triId = i;
                stat = 1;
            }
        }
    }
    return stat;
}

__device__ int intersectLeafTrianglesG(float & rayLength,
                        int & triId,
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
                triId = iElement;
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

template<int IsOrthographic> 
__device__ void computeEyeRay(Ray & eyeRay, 
                            uint x, uint y)
{
    const float u = ((float)x / (float)c_imageSize.x) - .5f;
    const float v = ((float)y / (float)c_imageSize.y) - .5f;
    
    if(IsOrthographic) {
        eyeRay.o = make_float4(u * c_cameraProp.x, v * c_cameraProp.x/c_cameraProp.y, 0.f, 1.f);
        eyeRay.d = make_float4(0.f, 0.f, -1.f, 0.f);
    }
    else {
        eyeRay.o = make_float4(0.f, 0.f, 0.f, 1.f);
        eyeRay.d = make_float4(u * c_cameraProp.x, v * c_cameraProp.x/c_cameraProp.y, -1.f, 0.f);
    }
    eyeRay.o = transform(c_modelViewMatrix, eyeRay.o);
    eyeRay.d = transform(c_modelViewMatrix, eyeRay.d);
    normalize(eyeRay.d);
}

template<int NumThreads, int IsOrthographic>
__global__ void renderImage_kernel(float4 * pix,
                int2 * internalNodeChildIndices,
				Aabb * internalNodeAabbs,
				KeyValuePair * elementHash,
				int4 * elementVertices,
				float3 * elementPoints)
{
    int *sdata = SharedMemory<int>();
    
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= c_imageSize.x) || (y >= c_imageSize.y)) return;
    
    Ray eyeRay;
    computeEyeRay<IsOrthographic>(eyeRay, x, y);
  
    float4 outRgbz = pix[y * c_imageSize.x + x];
    float rayLength = outRgbz.w;
/* 
 *  smem layout in ints
 *  n as num threads    64
 *  m as max stack size 64
 *  c as tri cache size 128
 *
 *  0   -> 1      stackSize
 *  4   -> 4+m-1       stack
 *  4+m -> 4+m+n-1  branching
 *  4+m+n -> 4+m+n+n-1  visiting
 *  4+m+n+n -> 4+m+n+n+12*c-1 triangle points
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
    int * sstack =  &sdata[4];
    int * sbranch = &sdata[4 + ADE_RAYTRAVERSE_MAX_STACK_SIZE];
    int * svisit =  &sdata[4 + ADE_RAYTRAVERSE_MAX_STACK_SIZE + NumThreads];
    float3 * strianglePCache = (float3 *)&sdata[4 + ADE_RAYTRAVERSE_MAX_STACK_SIZE + NumThreads + NumThreads];
    const uint tid = tId();
    if(tid<1) {
        sstack[0] = 0x80000000;
        sstackSize = 1;
    }
    __syncthreads();

    float3 hitP, hitN;
    int hitTriangle;
    int4 triV;
    int canLeafFitInSmem;
    int numTriangles;
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
// load triangles into smem
            numTriangles = child.y - child.x + 1;
            canLeafFitInSmem = (numTriangles <= ADE_RAYTRAVERSE_TRIANGLE_CACHE_SIZE);
            
            if(canLeafFitInSmem) 
                putLeafTrianglesInSmem<NumThreads>(strianglePCache,
                                            tid,
                                            numTriangles,
                                            child,
                                            elementHash,
                                            elementVertices,
                                            elementPoints);
            __syncthreads();
            
            leftBox = internalNodeAabbs[iNode];
            b1 = ray_box(lambda1, lambda2,
                    eyeRay,
                    rayLength,
                    leftBox);
            if(b1) {
// intersect triangles in leaf  
                if(canLeafFitInSmem) 
                    b1 = intersectLeafTrianglesS(rayLength,
                                hitTriangle,
                                eyeRay, 
                                numTriangles,
                                strianglePCache);
                else 
                    b1 = intersectLeafTrianglesG(rayLength,
                                hitTriangle,
                                eyeRay, 
                                child,
                                elementHash,
                                elementVertices,
                                elementPoints);
                if(b1) {
                    ray_progress(hitP, eyeRay, rayLength);
                    
                    if(canLeafFitInSmem) {
                        triangle_normal(hitN, strianglePCache[hitTriangle * 3],
                                    strianglePCache[hitTriangle * 3+1],
                                    strianglePCache[hitTriangle * 3+2]);
                    }
                    else {
                        triV = elementVertices[hitTriangle];
                        triangle_normal(hitN, elementPoints[triV.x],
                                    elementPoints[triV.y],
                                    elementPoints[triV.z]);
                    }
                    
                    hitN = float3_normalize(hitN);
                    outRgbz.x = hitN.x;
                    outRgbz.y = hitN.y;
                    outRgbz.z = hitN.z;
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
    pix[y * c_imageSize.x + x] = outRgbz;
}

