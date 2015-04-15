#ifndef FEMTETRAHEDRONSYSTEM_H
#define FEMTETRAHEDRONSYSTEM_H

#include <CudaTetrahedronSystem.h>
class CUDABuffer;
class FEMTetrahedronSystem : public CudaTetrahedronSystem {
public:
    FEMTetrahedronSystem();
    virtual ~FEMTetrahedronSystem();
    virtual void initOnDevice();
    
    void resetOrientation();
    void updateOrientation();
protected:

private:
    CUDABuffer * m_Re;
};

#endif        //  #ifndef FEMTETRAHEDRONSYSTEM_H

