#ifndef SAHTETRAHEDRONSYSTEM_H
#define SAHTETRAHEDRONSYSTEM_H

#include <CudaTetrahedronSystem.h>

class BaseBuffer;
class CUDABuffer;

class SahTetrahedronSystem : public CudaTetrahedronSystem
{
public:
    SahTetrahedronSystem();
    virtual ~SahTetrahedronSystem();
	virtual void initOnDevice();
    
protected:

private:
    
};

#endif        //  #ifndef SAHTETRAHEDRONSYSTEM_H

