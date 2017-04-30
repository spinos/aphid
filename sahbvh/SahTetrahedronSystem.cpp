#include "SahTetrahedronSystem.h"
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include <QuickSort.h>
#include <CudaDbgLog.h>
#include <boost/format.hpp>

SahTetrahedronSystem::SahTetrahedronSystem() 
{
}

SahTetrahedronSystem::~SahTetrahedronSystem() 
{
}

void SahTetrahedronSystem::initOnDevice()
{
    CudaTetrahedronSystem::initOnDevice();
}

