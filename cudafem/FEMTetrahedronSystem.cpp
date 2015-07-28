#include "FEMTetrahedronSystem.h"
#include <BaseBuffer.h>
#include <CUDABuffer.h>
#include <CudaCSRMatrix.h>
#include <cuFemTetrahedron_implement.h>
#include <QuickSort.h>
#include <CudaDbgLog.h>
#include <boost/format.hpp>
#include <FemGlobal.h>
#include <CudaBase.h>
#include <ATetrahedronMeshGroup.h>
#include <masssystem_impl.h>

CudaDbgLog bglg("stiffness.txt");
SplineMap1D FEMTetrahedronSystem::SplineMap;
float FEMTetrahedronSystem::YoungsModulus = 160000.f;
bool FEMTetrahedronSystem::NeedElasticity = true;
bool FEMTetrahedronSystem::NeedMass = false;
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
	m_BVolume = new CUDABuffer;
    m_stripeAttenuate = new CUDABuffer;
	m_tetrahedronElasticity = new CUDABuffer;
	m_hasBVolume = false;
}

FEMTetrahedronSystem::FEMTetrahedronSystem(ATetrahedronMeshGroup * md) :
BvhTetrahedronSystem(md)
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
	m_BVolume = new CUDABuffer;
    m_stripeAttenuate = new CUDABuffer;
	m_tetrahedronElasticity = new CUDABuffer;
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
	delete m_BVolume;
    delete m_stripeAttenuate;
	delete m_tetrahedronElasticity;
}

void FEMTetrahedronSystem::initOnDevice()
{
    m_Re->create(numTetrahedrons() * 36);
    createStiffnessMatrix();
	createVertexTetraHash();
    m_stiffnessMatrix->initOnDevice();
    
    m_deviceStiffnessTetraHash->create(numTetrahedrons() * 16 * 8);
    m_deviceStiffnessInd->create(m_stiffnessMatrix->numNonZero() * 4);
    m_deviceVertexTetraHash->create(numTetrahedrons() * 16 * 8);
    m_deviceVertexInd->create(numPoints() * 4);
    m_F0->create(numPoints() * 12);
    m_Fe->create(numPoints() * 12);
	m_BVolume->create(numTetrahedrons() * 64);
    m_stripeAttenuate->create(numTetrahedrons() * 4);
    m_stripeAttenuate->hostToDevice(hostElementValue(), numTetrahedrons() * 4);
	m_tetrahedronElasticity->create(numTetrahedrons() * 16);
    
    m_deviceStiffnessTetraHash->hostToDevice(m_stiffnessTetraHash->data());
    m_deviceStiffnessInd->hostToDevice(m_stiffnessInd->data());
    m_deviceVertexTetraHash->hostToDevice(m_vertexTetraHash->data());
    m_deviceVertexInd->hostToDevice(m_vertexInd->data());
    
    setDimension(numPoints());
    CudaConjugateGradientSolver::initOnDevice();
    BvhTetrahedronSystem::initOnDevice();
    m_hasBVolume = false;
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

unsigned combineKijt(unsigned k, unsigned i, unsigned j, unsigned t)
{
    return ( k<<5 | ( i<<3 | (j<<1 | t) ) );
}

uint64 upsample(uint a, uint b) 
{ return ((uint64)a << 32) | (uint64)b; }

void extractKij(unsigned c, unsigned & k, unsigned & i, unsigned & j)
{
    k = c>>5;
    i = (c & 31)>>3;
    j = c&3;
}

void extractKijt(unsigned c, unsigned & k, unsigned & i, unsigned & j, unsigned & t)
{
    k = c>>5;
    i = (c & 31)>>3;
    j = (c & 7)>>1;
	t = c & 1;
}


void FEMTetrahedronSystem::createStiffnessMatrix()
{
    CSRMap vertexConnection;
    unsigned *ind = hostTetrahedronIndices();
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
// i row j col
                    vertexConnection[matrixCoord(ind, k, w, i, j)] = 1;
                    if(j > i) {
// j row i col
                        vertexConnection[matrixCoord(ind, k, w, j, i)] = 1;
                    }
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
	
	QuickSortPair<uint64, uint> * kkij = new QuickSortPair<uint64, uint>[hashSize];

    for(k=0; k < n; k++) {
        for(i=0; i< 4; i++) {
            for(j=0; j<4; j++) {
                if(j >= i) {
                    kkij->key = upsample(vertexConnection[matrixCoord(ind, k, w, i, j)], k);
                    kkij->value = combineKijt(k,i,j, 0);
                    kkij++;
                    
                    if(j > i) {
                        kkij->key = upsample(vertexConnection[matrixCoord(ind, k, w, j, i)], k);
                        kkij->value = combineKijt(k,i,j, 1);
                        kkij++;
                    } 
                }
            }
        }
    }
    
    kkij -= hashSize;
    QuickSort1::Sort<uint64, uint>(kkij, 0, hashSize -1);
    
    m_stiffnessTetraHash->create(hashSize * 8);
    KeyValuePair * stiffnessTetraHash = (KeyValuePair *)m_stiffnessTetraHash->data();
   
    for(i=0; i< hashSize; i++) {
        stiffnessTetraHash[i].key = kkij[i].key >> 32;
        stiffnessTetraHash[i].value = kkij[i].value;
    }
    
    delete[] kkij;

    m_stiffnessInd->create(m_stiffnessMatrix->numNonZero() * 4);
    unsigned * stiffnessInd = (unsigned *)m_stiffnessInd->data();

    unsigned lastK = n + 2;
    for(i=0; i< hashSize; i++) {
        if(stiffnessTetraHash[i].key!= lastK) {
            lastK = stiffnessTetraHash[i].key;
            *stiffnessInd = i;
            stiffnessInd++;
        }
    }
}

void FEMTetrahedronSystem::createVertexTetraHash()
{
	unsigned *ind = hostTetrahedronIndices();
    unsigned i, j, k;
    const unsigned n = numTetrahedrons();
    const unsigned w = numPoints();
    const unsigned hashSize = n * 16;
	
	QuickSortPair<uint64, uint> * vkij = new QuickSortPair<uint64, uint>[hashSize];
	for(k=0; k < n; k++) {
        for(i=0; i< 4; i++) {
            for(j=0; j<4; j++) {
                
                vkij->key= upsample(ind[k*4+i], k);
                vkij->value=combineKij(k,i,j);
                vkij++;
                
            }
        }
    }
	
	vkij -= hashSize;
	
	QuickSort1::Sort<uint64, uint>(vkij, 0, hashSize -1);
    
	m_vertexTetraHash->create(hashSize * 8);
    KeyValuePair * vertexTetraHash = (KeyValuePair *)m_vertexTetraHash->data();

    for(i=0; i< hashSize; i++) {
        vertexTetraHash[i].key = vkij[i].key >> 32;
        vertexTetraHash[i].value = vkij[i].value;
    }
	
	delete[] vkij;
	
	m_vertexInd->create(w * 4);
    unsigned * vertexInd = (unsigned *)m_vertexInd->data();
    
    unsigned lastV = w + 2;
    for(i=0; i< hashSize; i++) {
        if(vertexTetraHash[i].key != lastV) {
            lastV = vertexTetraHash[i].key;
            *vertexInd = i;
            vertexInd++;
        }
    }
}

#define PRINT_RE 0
#define PRINT_K 0
#define PRINT_F0 0
#define PRINT_FE 0
#define PRINT_VETKIJIND 0
#define PRINT_VETKIJHSH 0
#define PRINT_STIFKIJIND 0
#define PRINT_STIFKIJHSH 0
#define PRINT_RHS 0
#define PRINT_ANCHOR 0
#define PRINT_SOLVERR 0

void FEMTetrahedronSystem::verbose()
{    
#if PRINT_VETKIJHSH
	bglg.writeHash(m_deviceVertexTetraHash, 
					numTetrahedrons() * 16, 
					" VkijHash ", CudaDbgLog::FOnce);
#endif	
#if PRINT_VETKIJIND
	bglg.writeUInt(m_deviceVertexInd, 
					numPoints(), 
					" VkijInd ", CudaDbgLog::FOnce);
#endif			
#if PRINT_STIFKIJHSH
	// bglg.writeHash(m_deviceStiffnessTetraHash, 
	//					numTetrahedrons() * 16, 
	//				" SkijHash ", CudaDbgLog::FOnce);
					
	const unsigned n = numTetrahedrons();
    bglg.write(boost::str(boost::format("\n stiffness-to-kij hash[%1%]\n") % (n * 16)));
	
    KeyValuePair * sth = (KeyValuePair *)m_stiffnessTetraHash->data();
    
    unsigned i, j, k, h, t;
    for(h=0; h< n * 16; h++) {
        extractKijt(sth[h].value, k, i, j, t);
		bglg.write(h);
		bglg.write(boost::str(boost::format("%1%:(%2%,%3%,%4%,%5%)\n") % sth[h].key %  k % i % j % t));
    }
#endif
#if PRINT_STIFKIJHSH
	bglg.writeUInt(m_deviceStiffnessInd, 
					m_stiffnessMatrix->numNonZero(), 
					" SkijInd ", CudaDbgLog::FOnce);
#endif
#if PRINT_RE	
	bglg.writeMat33(m_Re, 
					numTetrahedrons(), 
					" Re ", CudaDbgLog::FAlways);
#endif
#if PRINT_K
	bglg.writeMat33(m_stiffnessMatrix->valueBuf(), 
					m_stiffnessMatrix->numNonZero(), 
					" K ", CudaDbgLog::FAlways);
#endif
#if PRINT_F0
	bglg.writeVec3(m_F0, 
					numPoints(), 
					" F0 ", CudaDbgLog::FAlways);
#endif
#if PRINT_FE
	bglg.writeVec3(m_Fe, 
					numPoints(), 
					" Fe ", CudaDbgLog::FAlways);
#endif
#if PRINT_RHS
	bglg.writeVec3(rightHandSideBuf(), 
					numPoints(), 
					" Rhs ", CudaDbgLog::FAlways);
#endif
#if PRINT_ANCHOR
	bglg.writeUInt(anchorBuf(), 
					numPoints(), 
					" Anchor ", CudaDbgLog::FOnce);
#endif
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
	void * bv = m_BVolume->bufferOnDevice();
	tetrahedronfem::calculateRe((mat33 *)re, 
	                            (float3 *)x, 
	                            (float3 *)xi,
	                            (uint4 *)ind,
								(float4 *)bv,
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
    void * bv = m_BVolume->bufferOnDevice();
	void * e = m_tetrahedronElasticity->bufferOnDevice();
    tetrahedronfem::stiffnessAssembly((mat33 *)dst,
                                        (float3 *)xi,
                                        (uint4 *)tetv,
										(float4 *)bv,
                                        (mat33 *)re,
                                        (KeyValuePair *)sth,
                                        (uint *)ind,
                                        (float4 *)e,
                                        numTetrahedrons() * 16,
                                        m_stiffnessMatrix->numNonZero());
}

void FEMTetrahedronSystem::resetForce()
{
    void * d = m_F0->bufferOnDevice();
    cuFemTetrahedron_resetForce((float3 *)d, numPoints());
}

void FEMTetrahedronSystem::updateBVolume()
{
	if(m_hasBVolume) return;
	void * xi = deviceXi();
	void * tetv = deviceTretradhedronIndices();
	tetrahedronfem::computeBVolume((float4 * )m_BVolume->bufferOnDevice(), 
                    (float3 *)xi,
                    (uint4 * )tetv,
                    numTetrahedrons());
	m_hasBVolume = true;
}

void FEMTetrahedronSystem::updateMass()
{
	if(!NeedMass) return;
	
	void * mass = deviceMass();
	void * mass0 = deviceInitialMass();
	void * a = deviceAnchor();
	float relDensity = Density / 100.f;
	masssystem::computeMass((float * )mass, 
                    (float *)mass0,
					(uint *)a,
                    relDensity,
                    numPoints());
	CudaBase::CheckCudaError("fem update mass");
	NeedMass = false;
}

void FEMTetrahedronSystem::updateElasticity()
{
	if(!NeedElasticity) return;
	float bezierPnts[8];
	bezierPnts[0] = SplineMap.spline()->cv[0].x;
	bezierPnts[1] = SplineMap.spline()->cv[0].y;
	bezierPnts[2] = SplineMap.spline()->cv[1].x;
	bezierPnts[3] = SplineMap.spline()->cv[1].y;
	bezierPnts[4] = SplineMap.spline()->cv[2].x;
	bezierPnts[5] = SplineMap.spline()->cv[2].y;
	bezierPnts[6] = SplineMap.spline()->cv[3].x;
	bezierPnts[7] = SplineMap.spline()->cv[3].y;
	
	void * d = m_tetrahedronElasticity->bufferOnDevice();
	float * stripex = (float *)m_stripeAttenuate->bufferOnDevice();
	tetrahedronfem::computeElasticity((float4 *)d,
												stripex,
												YoungsModulus,
												numTetrahedrons(),
												bezierPnts);
	CudaBase::CheckCudaError("fem update elasticity");
	NeedElasticity = false;
}

void FEMTetrahedronSystem::updateForce()
{
    void * d = m_F0->bufferOnDevice();
    void * re = m_Re->bufferOnDevice();
    void * vth = m_deviceVertexTetraHash->bufferOnDevice();
    void * ind = m_deviceVertexInd->bufferOnDevice();
    void * xi = deviceXi();
    void * tetv = deviceTretradhedronIndices();
	void * bv = m_BVolume->bufferOnDevice();
	void * e = m_tetrahedronElasticity->bufferOnDevice();
    tetrahedronfem::internalForce((float3 *)d,
                                        (float3 *)xi,
                                        (uint4 *)tetv,
										(float4 *)bv,
                                        (mat33 *)re,
                                        (KeyValuePair *)vth,
                                        (unsigned *)ind,
										(float4 *)e,
                                        numTetrahedrons() * 16,
                                        numPoints());
    CudaBase::CheckCudaError("fem update force");
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
	cuFemTetrahedron_computeRhs((float3 *)rightHandSide(),
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
    float ferr;
	solve(deviceV(), m_stiffnessMatrix,
                deviceAnchor(), &ferr);
#if PRINT_SOLVERR
	bglg.write(boost::str(boost::format("cg solve error: %1%\n") % ferr));
#endif
}

void FEMTetrahedronSystem::integrate(float dt)
{
    tetrahedronfem::integrate((float3 *)deviceX(), 
								(float3 *)deviceV(), 
                                (float3 *)deviceVa(),
								(uint *)deviceAnchor(),
								dt, 
								numPoints());
    CudaBase::CheckCudaError("fem tetrahedron system integrate");
    
    //bglg.writeVec3(deviceAnchoredVBuf(), 
	//				numPoints(), 
	//				" Va ", CudaDbgLog::FAlways);
}

void FEMTetrahedronSystem::update()
{ 
#if DISABLE_FEM
	return CudaTetrahedronSystem::update();
#endif
	updateMass();
	updateExternalForce();
	updateBVolume();
	updateOrientation();
	updateElasticity();
	updateForce();
	updateStiffnessMatrix();
	dynamicsAssembly(1.f/60.f);
	solveConjugateGradient();
	BvhTetrahedronSystem::update();
}

void FEMTetrahedronSystem::SetNeedElasticity()
{ NeedElasticity = true; }

void FEMTetrahedronSystem::SetNeedMass()
{ NeedMass = true; }
//:~
