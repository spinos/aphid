#include "FEMTetrahedronSystem.h"
#include <CUDABuffer.h>
#include <CudaCSRMatrix.h>
#include <cuFemTetrahedron_implement.h>
#include <QuickSort.h>
#include <CudaDbgLog.h>

CudaDbgLog bglg("stiffness.txt");

FEMTetrahedronSystem::FEMTetrahedronSystem() 
{
    m_Re = new CUDABuffer;
    m_stiffnessMatrix = new CudaCSRMatrix;
    m_stiffnessTetraHash = new BaseBuffer;
    m_stiffnessInd = new BaseBuffer;
    m_vertexTetraHash = new BaseBuffer;
    m_vertexInd = new BaseBuffer;
    m_deviceStiffnessTetraHash = new CUDABuffer;
    m_deviceStiffnessInd = new CUDABuffer;
    m_deviceVertexTetraHash = new CUDABuffer;
    m_deviceVertexInd = new CUDABuffer;
    m_F0 = new CUDABuffer;
    m_Fe = new CUDABuffer;
}

FEMTetrahedronSystem::~FEMTetrahedronSystem() 
{
    delete m_Re;
    delete m_stiffnessMatrix;
    delete m_stiffnessTetraHash;
    delete m_stiffnessInd;
    delete m_vertexTetraHash;
    delete m_vertexInd;
    delete m_deviceStiffnessTetraHash;
    delete m_deviceStiffnessInd;
    delete m_deviceVertexTetraHash;
    delete m_deviceVertexInd;
    delete m_F0;
    delete m_Fe;
}

void FEMTetrahedronSystem::initOnDevice()
{
    m_Re->create(numTetrahedrons() * 36);
    createStiffnessMatrix();
    m_stiffnessMatrix->initOnDevice();
    
    m_deviceStiffnessTetraHash->create(numTetrahedrons() * 16 * 8);
    m_deviceStiffnessInd->create(m_stiffnessMatrix->numNonZero() * 4);
    m_deviceVertexTetraHash->create(numTetrahedrons() * 16 * 8);
    m_deviceVertexInd->create(numPoints() * 4);
    m_F0->create(numPoints() * 12);
    m_Fe->create(numPoints() * 12);
    
    m_deviceStiffnessTetraHash->hostToDevice(m_stiffnessTetraHash->data());
    m_deviceStiffnessInd->hostToDevice(m_stiffnessInd->data());
    m_deviceVertexTetraHash->hostToDevice(m_vertexTetraHash->data());
    m_deviceVertexInd->hostToDevice(m_vertexInd->data());
    
    setDimension(numPoints());
    CudaConjugateGradientSolver::initOnDevice();
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

uint64 upsample(uint a, uint b) 
{ return ((uint64)a << 32) | (uint64)b; }

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
    unsigned i, j, k, h;
    const unsigned n = numTetrahedrons();
    const unsigned w = numPoints();
    const unsigned hashSize = n * 16;
    
    QuickSortPair<uint64, uint> * vtkv = new QuickSortPair<uint64, uint>[hashSize];

    for(k=0; k < n; k++) {
        for(i=0; i< 4; i++) {
            h = ind[k*4+i];
            for(j=0; j<4; j++) {
                
                vtkv->key= upsample(h, k);
                vtkv->value=combineKij(k,i,j);
                vtkv++;
                
                if(j >= i) {
                    vertexConnection[matrixCoord(ind, k, w, i, j)] = 1;
                    if(j > i) {
                        vertexConnection[matrixCoord(ind, k, w, j, i)] = 1;
                    }
                }
            }
        }
    }
    
    vtkv -= hashSize;
    QuickSort1<uint64, uint>::Sort(vtkv, 0, hashSize -1);
    
    m_vertexTetraHash->create(hashSize * 8);
    KeyValuePair * vertexTetraHash = (KeyValuePair *)m_vertexTetraHash->data();

    for(i=0; i< hashSize; i++) {
        vertexTetraHash[i].key = vtkv[i].key >> 32;
        vertexTetraHash[i].value = vtkv[i].value;
    }

    CSRMap::iterator it = vertexConnection.begin();
    i = 0;
    for(;it!=vertexConnection.end();++it) {
        it->second = i;
        i++;
    }
    
    m_stiffnessMatrix->create(CSRMatrix::tMat33, w, vertexConnection);

    for(k=0; k < n; k++) {
        for(i=0; i< 4; i++) {
            for(j=0; j<4; j++) {
                if(j >= i) {
                    vtkv->key = upsample(vertexConnection[matrixCoord(ind, k, w, i, j)], k);
                    vtkv->value = combineKij(k,i,j);
                    vtkv++;
                    
                    if(j > i) {
                        vtkv->key = upsample(vertexConnection[matrixCoord(ind, k, w, j, i)], k);
                        vtkv->value = combineKij(k,i,j);
                        vtkv++;
                    } 
                }
            }
        }
    }
    
    vtkv -= hashSize;
    QuickSort1<uint64, uint>::Sort(vtkv, 0, hashSize -1);
    
    m_stiffnessTetraHash->create(hashSize * 8);
    KeyValuePair * stiffnessTetraHash = (KeyValuePair *)m_stiffnessTetraHash->data();
   
    for(i=0; i< hashSize; i++) {
        stiffnessTetraHash[i].key = vtkv[i].key >> 32;
        stiffnessTetraHash[i].value = vtkv[i].value;
    }
    
    delete[] vtkv;

    m_stiffnessInd->create(m_stiffnessMatrix->numNonZero() * 4);
    unsigned * scanned = (unsigned *)m_stiffnessInd->data();

// prefix sum to get element start
    unsigned lastK = n + 2;
    for(i=0; i< hashSize; i++) {
        if(stiffnessTetraHash[i].key!= lastK) {
            lastK = stiffnessTetraHash[i].key;
            *scanned = i;
            scanned++;
        }
    }
    
    m_vertexInd->create(w * 4);
    unsigned * vertexInd = (unsigned *)m_vertexInd->data();
    
    unsigned lastV = w + 2;
    for(i=0; i< hashSize; i++) {
        if(vertexTetraHash[i].key!= lastV) {
            lastV = vertexTetraHash[i].key;
            *vertexInd = i;
            vertexInd++;
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
    
    unsigned * ind = (unsigned *)m_stiffnessInd->data();
    std::cout<<"\n stiffness indirection["<<nnz<<"]:\n";
    for(i=0; i<nnz; i++)
        std::cout<<" "<<ind[i];
    
    KeyValuePair * vertexTetraHash = (KeyValuePair *)m_vertexTetraHash->data();
    std::cout<<"\n vertex-to_tetra hash["<<n * 16<<"]:\n";
    for(h=0; h<n*16; h++) {
        extractKij(vertexTetraHash[h].value, k, i, j);
        std::cout<<" v"<<vertexTetraHash[h].key<<":"<<k<<","<<i<<","<<j<<" ";
    }
    
    unsigned * vertexInd = (unsigned *)m_vertexInd->data();
    std::cout<<"\n vertex indirection["<<numPoints()<<"]\n";
    for(h=0; h<numPoints(); h++) {
        std::cout<<" "<<vertexInd[h]<<" ";
    }
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
    void * dst = m_stiffnessMatrix->deviceValue();
    cuFemTetrahedron_resetStiffnessMatrix((mat33 *)dst, 
                                        m_stiffnessMatrix->numNonZero());
}

void FEMTetrahedronSystem::updateStiffnessMatrix() 
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

void FEMTetrahedronSystem::resetForce()
{
    void * d = m_F0->bufferOnDevice();
    cuFemTetrahedron_resetForce((float3 *)d, numPoints());
}

void FEMTetrahedronSystem::updateForce()
{
    void * d = m_F0->bufferOnDevice();
    void * re = m_Re->bufferOnDevice();
    void * vth = m_deviceVertexTetraHash->bufferOnDevice();
    void * ind = m_deviceVertexInd->bufferOnDevice();
    void * xi = deviceXi();
    void * tetv = deviceTretradhedronIndices();
    cuFemTetrahedron_internalForce((float3 *)d,
                                        (float3 *)xi,
                                        (uint4 *)tetv,
                                        (mat33 *)re,
                                        (KeyValuePair *)vth,
                                        (unsigned *)ind,
                                        numTetrahedrons() * 16,
                                        numPoints());
}

void FEMTetrahedronSystem::dynamicsAssembly(float dt)
{
    void * X = deviceX();
	void * V = deviceV();
	void * mass = deviceMass();
	void * stiffness = m_stiffnessMatrix->deviceValue();
	void * rowPtr = m_stiffnessMatrix->deviceRowPtr();
	void * colInd = m_stiffnessMatrix->deviceColInd();
	void * f0 = m_F0->bufferOnDevice();
	void * fe = m_Fe->bufferOnDevice();
	cuFemTetrahedron_computeRhsA((float3 *)rightHandSide(),
                                (float3 *)X,
                                (float3 *)V,
                                (float *)mass,
                                (mat33 *)stiffness,
                                (uint *)rowPtr,
                                (uint *)colInd,
                                (float3 *)f0,
								(float3 *)fe,
                                dt,
                                numPoints());
}

void FEMTetrahedronSystem::updateExternalForce()
{
    void * force = m_Fe->bufferOnDevice();
    void * mass = deviceMass();
    cuFemTetrahedron_externalForce((float3 *)force,
                                (float *)mass,
                                numPoints());
}

void FEMTetrahedronSystem::solveConjugateGradient()
{
    return;
    float error;
	solve(deviceV(), m_stiffnessMatrix,
                deviceAnchor(), &error);
}

void FEMTetrahedronSystem::integrate(float dt)
{
    cuFemTetrahedron_integrate((float3 *)deviceX(), 
								(float3 *)deviceV(), 
								(uint *)deviceAnchor(),
								dt, 
								numPoints());
}

void FEMTetrahedronSystem::update()
{
	updateExternalForce();
	resetStiffnessMatrix();
	updateOrientation();
	updateStiffnessMatrix();
	dynamicsAssembly(1.f/60.f);
	solveConjugateGradient();
	
	bglg.writeMat33(m_Re, 
					numTetrahedrons(), 
					" Re ", CudaDbgLog::FAlways);
/*
	bglg.writeMat33(m_stiffnessMatrix->valueBuf(), 
					m_stiffnessMatrix->numNonZero(), 
					" K ", CudaDbgLog::FAlways);
*/					
	CudaTetrahedronSystem::update();
}
