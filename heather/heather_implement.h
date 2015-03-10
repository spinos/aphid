#ifndef HEATHER_IMPLEMENT_H
#define HEATHER_IMPLEMENT_H
namespace CUU {
#include <cuda_runtime_api.h>
typedef unsigned int uint;
static uint iDivUp(uint dividend, uint divisor)
{
    return ( (dividend % divisor) == 0 ) ? (dividend / divisor) : (dividend / divisor + 1);
}
extern "C" {
void heatherFillImage( ushort4 * dstCol,  float * dstDep,  ushort4 * srcCol,  float * srcDep, uint numPix);
void heatherMixImage( ushort4 * dstCol,  float * dstDep,  ushort4 * srcCol,  float * srcDep, uint numPix);
}
}
#endif        //  #ifndef HEATHER_IMPLEMENT_H

