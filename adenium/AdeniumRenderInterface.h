#include <bvh_common.h>
#include <radixsort_implement.h>
namespace adetrace {

void setImageSize(int * src);

void setCameraProp(float * src);

void setModelViewMatrix(float * src, 
            uint size);

void resetImage(float4 * pix, 
            uint n);

void renderImage(float4 * pix,
                uint imageW,
                uint imageH,
                int2 * nodes,
				Aabb * nodeAabbs,
				KeyValuePair * elementHash,
				int4 * elementVertices,
				float3 * elementPoints,
				int isOrthographic);
}
