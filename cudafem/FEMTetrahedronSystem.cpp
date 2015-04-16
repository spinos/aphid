#include "FEMTetrahedronSystem.h"
#include <CUDABuffer.h>
#include <CudaCSRMatrix.h>
#include <cuFemTetrahedron_implement.h>

FEMTetrahedronSystem::FEMTetrahedronSystem() 
{
    m_Re = new CUDABuffer;
    m_stiffnessMatrix = new CudaCSRMatrix;
}

FEMTetrahedronSystem::~FEMTetrahedronSystem() 
{
    delete m_Re;
    delete m_stiffnessMatrix;
}

void FEMTetrahedronSystem::initOnDevice()
{
    m_Re->create(numTetrahedrons() * 36);
    createStiffnessMatrix();
    m_stiffnessMatrix->initOnDevice();
    CudaTetrahedronSystem::initOnDevice();
}

void FEMTetrahedronSystem::createStiffnessMatrix()
{
    CSRMap vertexConnection;
    unsigned *ind = hostTretradhedronIndices();
    unsigned i, j, k, v;
    const unsigned n = numTetrahedrons();
    const unsigned w = numPoints();
    for(k=0; k < n; k++) {
        for(i=0; i< 4; i++) {
            for(j=0; j<4; j++) {
                // std::cout<<" i"<<i<<" j"<<j<<"\n";
                if(j >= i) {
                    v = ind[k*4 + i];
                    // std::cout<<" row"<<v<<" col "<<ind[k*4 + j]<<"\n";
                    
                    vertexConnection[v * w + ind[k*4 + j]] = 1;
                    
                    if(j > i) {
                        v = ind[k*4 + j];
                        // std::cout<<" row"<<v<<" col "<<ind[k*4 + i]<<"\n";
                        
                        vertexConnection[v * w + ind[k*4 + i]] = 1;
                    } 
                }
            }
        }
    }
    
    m_stiffnessMatrix->create(CSRMatrix::tMat33, w, vertexConnection);
}

void FEMTetrahedronSystem::resetOrientation()
{
    void * d = m_Re->bufferOnDevice();
    cuFemTetrahedron_resetRe((mat33 *)d, numTetrahedrons());
}
    
void FEMTetrahedronSystem::updateOrientation()
{
    void * re = m_Re->bufferOnDevice();
    void * x = deviceX();
	void * xi = deviceXi();
	void * ind = deviceTretradhedronIndices();
	cuFemTetrahedron_calculateRe((mat33 *)re, 
	                            (float3 *)x, 
	                            (float3 *)xi,
	                            (uint4 *)ind,
	                            numTetrahedrons());
}
