#include "FEMTetrahedronSystem.h"
#include <CUDABuffer.h>
#include <CudaCSRMatrix.h>
#include <cuFemTetrahedron_implement.h>
#include <QuickSort.h>

FEMTetrahedronSystem::FEMTetrahedronSystem() 
{
    m_Re = new CUDABuffer;
    m_stiffnessMatrix = new CudaCSRMatrix;
    m_stiffnessTetraHash = new BaseBuffer;
    m_stiffnessInd = new BaseBuffer;
    m_deviceStiffnessTetraHash = new CUDABuffer;
    m_deviceStiffnessInd = new CUDABuffer;
}

FEMTetrahedronSystem::~FEMTetrahedronSystem() 
{
    delete m_Re;
    delete m_stiffnessMatrix;
    delete m_stiffnessTetraHash;
}

void FEMTetrahedronSystem::initOnDevice()
{
    m_Re->create(numTetrahedrons() * 36);
    createStiffnessMatrix();
    m_stiffnessMatrix->initOnDevice();
    
    m_deviceStiffnessTetraHash->create(numTetrahedrons() * 16 * 8);
    m_deviceStiffnessInd->create(m_stiffnessMatrix->numNonZero() * 4);
    
    m_deviceStiffnessTetraHash->hostToDevice(m_stiffnessTetraHash->data());
    m_deviceStiffnessInd->hostToDevice(m_stiffnessInd->data());
    
    CudaTetrahedronSystem::initOnDevice();
}

unsigned matrixCoord(unsigned * indices, unsigned tet, 
                        unsigned n, unsigned a, unsigned b)
{
    return indices[tet*4 + a] * n + indices[tet*4 + b];
}

unsigned combineKij(unsigned k, unsigned i, unsigned j)
{
    return (k<<5 | ( i<<3 | j));
}

void extractKij(unsigned c, unsigned & k, unsigned & i, unsigned & j)
{
    k = c>>5;
    i = (c & 31)>>3;
    j = c&3;
}

void FEMTetrahedronSystem::createStiffnessMatrix()
{
    CSRMap vertexConnection;
    unsigned *ind = hostTretradhedronIndices();
    unsigned i, j, k;
    const unsigned n = numTetrahedrons();
    const unsigned w = numPoints();
    for(k=0; k < n; k++) {
        for(i=0; i< 4; i++) {
            for(j=0; j<4; j++) {
                if(j >= i) {
                    vertexConnection[matrixCoord(ind, k, w, i, j)] = 1;
                    
                    if(j > i)
                        vertexConnection[matrixCoord(ind, k, w, j, i)] = 1; 
                }
            }
        }
    }
    
    CSRMap::iterator it = vertexConnection.begin();
    i = 0;
    for(;it!=vertexConnection.end();++it) {
        it->second = i;
        i++;
    }
    
    m_stiffnessMatrix->create(CSRMatrix::tMat33, w, vertexConnection);
    
    m_stiffnessTetraHash->create(n * 16 * 8);
    KeyValuePair * sth = (KeyValuePair *)m_stiffnessTetraHash->data();
    
    for(k=0; k < n; k++) {
        for(i=0; i< 4; i++) {
            for(j=0; j<4; j++) {
                if(j >= i) {
                    sth->key = vertexConnection[matrixCoord(ind, k, w, i, j)];
                    sth->value = combineKij(k,i,j);
                    sth++;
                    
                    if(j > i) {
                        sth->key = vertexConnection[matrixCoord(ind, k, w, j, i)];
                        sth->value = combineKij(k,i,j);
                        sth++;
                    } 
                }
            }
        }
    }
    
    sth -= n * 16;
   
    QuickSort::Sort((unsigned *)sth, 0, n * 16 -1);
    
    m_stiffnessInd->create(m_stiffnessMatrix->numNonZero() * 4);
    unsigned * scanned = (unsigned *)m_stiffnessInd->data();

// scan to get element start
    unsigned lastK = n + 2;
    for(i=0; i< n * 16; i++) {
        if(sth[i].key!= lastK) {
            lastK = sth[i].key;
            *scanned = i;
            scanned++;
        }
    }
}

void FEMTetrahedronSystem::verbose()
{
    std::cout<<"\n stiffness matrix:\n";
    m_stiffnessMatrix->verbose();
    
    const unsigned n = numTetrahedrons();
    
    std::cout<<"\n stiffness-to-tetra hash["<<n * 16<<"]: ";
    KeyValuePair * sth = (KeyValuePair *)m_stiffnessTetraHash->data();
    
    unsigned i, j, k, h;
    for(h=0; h< n * 16; h++) {
        extractKij(sth[h].value, k, i, j);
        std::cout<<" "<<sth[h].key<<":"<<k<<","<<i<<","<<j<<" ";
    }
    
    const unsigned nnz = m_stiffnessMatrix->numNonZero();
    
    std::cout<<"\n stiffness indirection["<<nnz<<"]: ";
    unsigned * ind = (unsigned *)m_stiffnessInd->data();
    for(i=0; i<nnz; i++)
        std::cout<<" "<<ind[i];
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

void FEMTetrahedronSystem::resetStiffnessMatrix()
{
    void * dst = m_stiffnessMatrix->valueBuf()->bufferOnDevice();
    cuFemTetrahedron_resetStiffnessMatrix((mat33 *)dst, 
                                        m_stiffnessMatrix->numNonZero());
}

void FEMTetrahedronSystem::stiffnessAssembly() 
{
    void * dst = m_stiffnessMatrix->valueBuf()->bufferOnDevice();
    void * re = m_Re->bufferOnDevice();
    void * sth = m_deviceStiffnessTetraHash->bufferOnDevice();
    void * ind = m_deviceStiffnessInd->bufferOnDevice();
    void * xi = deviceXi();
    void * tetv = deviceTretradhedronIndices();
    cuFemTetrahedron_stiffnessAssembly((mat33 *)dst,
                                        (float3 *)xi,
                                        (uint4 *)tetv,
                                        (mat33 *)re,
                                        (KeyValuePair *)sth,
                                        (unsigned *)ind,
                                        numTetrahedrons() * 16,
                                        m_stiffnessMatrix->numNonZero());
}

