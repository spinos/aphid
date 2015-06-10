#ifndef BVHTTRIANGLESYSTEM_H
#define BVHTTRIANGLESYSTEM_H
#include "CudaLinearBvh.h"
#include "TriangleSystem.h"
class CUDABuffer;

class BvhTriangleSystem : public TriangleSystem, public CudaLinearBvh {
public:
    BvhTriangleSystem(ATriangleMesh * md);
    virtual ~BvhTriangleSystem();

    virtual void initOnDevice();
	virtual void update();
	
protected:

private:
	void formTetrahedronAabbs();
private:

};

#endif        //  #ifndef BVHTTRIANGLESYSTEM_H

