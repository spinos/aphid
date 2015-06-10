#ifndef CUDATRIANGLESYSTEM_H
#define CUDATRIANGLESYSTEM_H
#include "CudaLinearBvh.h"
#include "TriangleSystem.h"
class CUDABuffer;

class CudaTriangleSystem : public TriangleSystem, public CudaLinearBvh {
public:
    CudaTriangleSystem(ATriangleMesh * md);
    virtual ~CudaTriangleSystem();

    virtual void initOnDevice();
	virtual void update();
	
protected:

private:
	void formTetrahedronAabbs();
private:

};

#endif        //  #ifndef CUDATRIANGLESYSTEM_H

