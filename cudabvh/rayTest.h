#ifndef RAY_TEST_H
#define RAY_TEST_H

#include "bvh_common.h"

class CudaLinearBvh;
class CUDABuffer;
class BaseBuffer;

class RayTest
{
public:
	RayTest();
	virtual ~RayTest();
	
	void setBvh(CudaLinearBvh * bvh);
	void createRays(uint m, uint n);
	
	const unsigned numRays() const;
	
	void setAlpha(float x);
	RayInfo * getRays();
	
	void update();
	
protected:
    
private:
	void formRays();
	void rayTraverse();
	
private:
	CUDABuffer * m_rays;
	CudaLinearBvh * m_bvh;
    BaseBuffer * m_displayRays;
	
	unsigned m_numRays, m_rayDim;
	float m_alpha;
};
#endif        //  #ifndef RAY_TEST_H

