#include "FEMTetrahedronSystem.h"
#include <CUDABuffer.h>
#include <cuFemTetrahedron_implement.h>
FEMTetrahedronSystem::FEMTetrahedronSystem() 
{
    m_Re = new CUDABuffer;
}

FEMTetrahedronSystem::~FEMTetrahedronSystem() {}

void FEMTetrahedronSystem::initOnDevice()
{
    m_Re->create(numTetradedrons() * 36);
    CudaTetrahedronSystem::initOnDevice();
}

void FEMTetrahedronSystem::resetOrientation()
{
    void * d = m_Re->bufferOnDevice();
    cuFemTetrahedron_resetRe((mat33 *)d, numTetradedrons());
}
    
void FEMTetrahedronSystem::updateOrientation()
{
    
}
