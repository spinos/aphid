#include "matrix_math.cuh"
#include "ray_intersection.cuh"
#define ADE_RAYTRAVERSE_MAX_STACK_SIZE 64
#define ADE_RAYTRAVERSE_MAX_STACK_SIZE_M_2 62

__constant__ mat44 c_modelViewMatrix;  // inverse view matrix

inline __device__ int isStackFull(int stackSize)
{return stackSize > ADE_RAYTRAVERSE_MAX_STACK_SIZE_M_2; }

inline __device__ int outOfStack(int stackSize)
{return (stackSize < 1 || stackSize > ADE_RAYTRAVERSE_MAX_STACK_SIZE); }

inline __device__ int isInternalNode(int2 child)
{ return (child.x>>31) != 0; }

__global__ void resetImage_kernel(float4 * pix, 
								uint maxInd)
{
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= maxInd) return;
	pix[ind] = make_float4(0.f, 0.f, 0.f, 1e20f);
}

__global__ void renderImageOrthographic_kernel(float4 * pix,
                uint imageW,
                uint imageH,
                float fovWidth,
                float aspectRatio,
                int2 * internalNodeChildIndices,
				Aabb * internalNodeAabbs)
{
    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= imageW) || (y >= imageH)) return;
    
    float u = (x / (float) imageW) - .5f;
    float v = (y / (float) imageH) - .5f;
    
    Ray eyeRay;
    eyeRay.o = make_float4(u * fovWidth, v * fovWidth/aspectRatio, 0.f, 1.f);
    eyeRay.d = make_float4(0.f, 0.f, -1000.f, 0.f);
    
    eyeRay.o = transform(c_modelViewMatrix, eyeRay.o);
    eyeRay.d = transform(c_modelViewMatrix, eyeRay.d);
    normalize(eyeRay.d);
    
    int stack[ADE_RAYTRAVERSE_MAX_STACK_SIZE];
	int stackSize = 1;
	stack[0] = 0x80000000;
	
    int isInternal;
    int iNode;
    int2 child;
    Aabb leftBox, rightBox;
    float lambda1, lambda2;
    float mu1, mu2;
    int b1, b2;
    float4 outRgbz = pix[y * imageW + x];
    float rayLength = outRgbz.w;
    for(;;) {
        if(outOfStack(stackSize)) break;
        iNode = stack[ stackSize - 1 ];
		stackSize--;
		
		iNode = getIndexWithInternalNodeMarkerRemoved(iNode);
        child = internalNodeChildIndices[iNode];
        isInternal = isInternalNode(child);
		
        if(isInternal) {
            if(isStackFull(stackSize)) continue;
            
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
            
            if(b1 > 0 && b2 > 0) { 
// visit both children
                if(mu1 < lambda1) {
// vist right child first
                    stack[ stackSize ] = child.x;
                    stackSize++;
                    stack[ stackSize ] = child.y;
                    stackSize++;
                }
                else {
// vist left child first 
                    stack[ stackSize ] = child.y;
                    stackSize++;
                    stack[ stackSize ] = child.x;
                    stackSize++;
                }
            }
            else if(b1 > 0) { 
// visit left child
                stack[ stackSize ] = child.x;
                stackSize++;
            }
            else if(b2 > 0) { 
// visit right child
                stack[ stackSize ] = child.y;
                stackSize++;
            }
            else { 
// visit no child
            }
        }
        else {
// todo intersect triangles in leaf
            leftBox = internalNodeAabbs[iNode];
            b1 = ray_box(lambda1, lambda2,
                    eyeRay,
                    rayLength,
                    leftBox);
            rayLength = lambda2;
            outRgbz.x = rayLength/300.f;
            outRgbz.y = rayLength/300.f;
            outRgbz.z = rayLength/300.f;
            outRgbz.w = rayLength;
        }
    }
    pix[y * imageW + x] = outRgbz;
}

