#ifndef FEMTETRAHEDRONSYSTEM_H
#define FEMTETRAHEDRONSYSTEM_H

#include <CudaTetrahedronSystem.h>
class CUDABuffer;
class CSRMatrix;
class FEMTetrahedronSystem : public CudaTetrahedronSystem {
public:
    FEMTetrahedronSystem();
    virtual ~FEMTetrahedronSystem();
    virtual void initOnDevice();
    
    void resetOrientation();
    void updateOrientation();
protected:
    void createStiffnessMatrix();
    
private:
    CUDABuffer * m_Re;
    CSRMatrix * m_stiffnessMatrix;
};

#endif        //  #ifndef FEMTETRAHEDRONSYSTEM_H

