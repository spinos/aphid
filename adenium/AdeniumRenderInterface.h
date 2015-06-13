#include <bvh_common.h>
#include <radixsort_implement.h>
namespace adetrace {
    
void setModelViewMatrix(float * src, 
            uint size);

void resetImage(float4 * pix, 
            uint n);

void renderImageOrthographic(float4 * pix,
                uint imageW,
                uint imageH,
                float fovWidth,
                float aspectRatio,
				int2 * nodes,
				Aabb * nodeAabbs,
				KeyValuePair * elementHash,
				int4 * elementVertices,
				float3 * elementPoints);
}
