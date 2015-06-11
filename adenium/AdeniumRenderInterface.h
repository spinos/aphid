#include <bvh_common.h>

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
				Aabb * nodeAabbs);
}
