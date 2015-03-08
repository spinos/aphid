#include "ScanUtil.h"
#include "CUDABuffer.h"
ScanUtil::ScanUtil() {}

unsigned ScanUtil::getScanResult(CUDABuffer * counts, 
                                CUDABuffer * sums, 
                                unsigned bufferLength)
{
    unsigned a, b;
    counts->deviceToHost(&a, 4*(bufferLength - 1), 4);
    sums->deviceToHost(&b, 4*(bufferLength - 1), 4);
    return a + b;
};
