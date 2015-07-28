#ifndef FEMTETRAHEDRONSYSTEM_H
#define FEMTETRAHEDRONSYSTEM_H

#include <BvhTetrahedronSystem.h>
#include <CudaConjugateGradientSolver.h>
#include <SplineMap1D.h>
class BaseBuffer;
class CUDABuffer;
class CudaCSRMatrix;
class ATetrahedronMeshGroup;

class FEMTetrahedronSystem : public BvhTetrahedronSystem, 
                                public CudaConjugateGradientSolver
{
public:
    FEMTetrahedronSystem();
    FEMTetrahedronSystem(ATetrahedronMeshGroup * md);
    virtual ~FEMTetrahedronSystem();
	virtual void initOnDevice();
	virtual void update();
// override BvhTetrahedronSystem
    virtual void integrate(float dt);
	static SplineMap1D SplineMap;
	static float YoungsModulus;
    static void SetNeedElasticity();
	static void SetNeedMass();
    void verbose();
protected:
// override cudamasssystem
    virtual void updateMass();
private:
    void createStiffnessMatrix();
	void createVertexTetraHash();
    void resetOrientation();
    void updateOrientation();
	void updateElasticity();
    void resetStiffnessMatrix();
    void updateStiffnessMatrix();
    void resetForce();
    void updateForce();
    void dynamicsAssembly(float dt);
    void updateExternalForce();
	void solveConjugateGradient();
	void updateBVolume();
private:
    CUDABuffer * m_Re;
    CudaCSRMatrix * m_stiffnessMatrix;
    BaseBuffer * m_stiffnessTetraHash;
    BaseBuffer * m_stiffnessInd;
    BaseBuffer * m_vertexTetraHash;
    BaseBuffer * m_vertexInd;
    CUDABuffer * m_deviceStiffnessTetraHash;
    CUDABuffer * m_deviceStiffnessInd;
    CUDABuffer * m_deviceVertexTetraHash;
    CUDABuffer * m_deviceVertexInd;
    CUDABuffer * m_F0;
    CUDABuffer * m_Fe;
	CUDABuffer * m_BVolume;
    CUDABuffer * m_stripeAttenuate;
	CUDABuffer * m_tetrahedronElasticity;
    bool m_hasBVolume;
	static bool NeedElasticity;
	static bool NeedMass;
};

#endif        //  #ifndef FEMTETRAHEDRONSYSTEM_H

