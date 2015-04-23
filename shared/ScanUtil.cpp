#include "ScanUtil.h"
#include "CUDABuffer.h"
#include <iostream>
ScanUtil::ScanUtil() {}

unsigned ScanUtil::getScanResult(CUDABuffer * counts, 
                                CUDABuffer * sums, 
                                unsigned bufferLength)
{
    unsigned a=0, b=0;
    counts->deviceToHost(&a, 4*(bufferLength - 1), 4);
    sums->deviceToHost(&b, 4*(bufferLength - 1), 4);
    return a + b;
};
