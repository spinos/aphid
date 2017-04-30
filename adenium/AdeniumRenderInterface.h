#include <bvh_common.h>
#include <radixsort_implement.h>
namespace adetrace {

void setImageSize(int * src);

void setCameraProp(float * src);

void setModelViewMatrix(float * src, 
            uint size);

void resetImage(uint * pix, 
				float * depth,
            uint n);

void renderImage(uint * pix,
                float * depth,
                uint imageW,
                uint imageH,
                int2 * nodes,
				Aabb * nodeAabbs,
				KeyValuePair * elementHash,
				int4 * elementVertices,
				float3 * elementPoints,
				int isOrthographic);
}
