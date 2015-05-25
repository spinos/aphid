#include <cuda_runtime_api.h>
#include "bvh_common.h"
#include "radixsort_implement.h"

namespace sahdecompress {
    
__global__ void initHash_kernel(KeyValuePair * primitiveIndirections,
                    uint n)
{
    uint ind = blockIdx.x*blockDim.x + threadIdx.x;
	if(ind >= n) return;
	
	primitiveIndirections[ind].value = ind;
}

}
