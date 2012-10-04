#include "cooktorrance_implement.h"

#include "brdf_common.h"

__global__ void 
cooktorrance_kernel(float3* pos, unsigned int width, float3 V, float3 N, float m, float f0, bool include_F, bool include_G)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    float3 L = calculateL(pos, width, x, y);
    float3 H = normalize(add(L, V));
    float NdotH = dot(N, H);
    float VdotH = dot(V, H);
    float NdotL = dot(N, L);
    float NdotV = dot(N, V);
    float oneOverNdotV = 1.0 / NdotV;
    
    float D = beckmann(m, NdotH);
    float F = fresnel(f0, VdotH);

    NdotH = NdotH + NdotH;
    
    float G = (NdotV < NdotL) ? 
        ((NdotV*NdotH < VdotH) ?
         NdotH / VdotH :
         oneOverNdotV)
        :
        ((NdotL*NdotH < VdotH) ?
         NdotH*NdotL / (VdotH*NdotV) :
         oneOverNdotV);

    if (include_G) G = oneOverNdotV;
    float val = NdotH < 0 ? 0.0 : D * G ;

    if (include_F) val *= F;
    
    pos[y*width+x] = scale(L, val);
}

extern "C" void cooktorrance_brdf(float3 *pos, unsigned numVertices, unsigned width, float3 V, float3 N, float m, float f0, bool include_F, bool include_G)
{
    dim3 block(8, 8, 1);
    unsigned height = numVertices / width;
    dim3 grid(width / block.x, height / block.y, 1);
    cooktorrance_kernel<<< grid, block>>>(pos, width, V, N, m, f0, include_F, include_G);
}
