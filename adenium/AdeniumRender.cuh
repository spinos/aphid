#include "matrix_math.cuh"

struct __align__(16) Ray {
	float4 o;	// origin
	float4 d;	// direction
};

__constant__ mat44 c_modelViewMatrix;  // inverse view matrix

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
                float aspectRatio)
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
    
    pix[y * imageW + x] = eyeRay.d;
}

