#ifndef RAY_TEST_H
#define RAY_TEST_H

class CudaLinearBvh;
class CUDABuffer;

class RayTest
{
public:
	RayTest();
	virtual ~RayTest();
	
	void setBvh(CudaLinearBvh * bvh);
	void createRays(uint m, uint n);
	
	const unsigned numRays() const;
	
	void setAlpha(float x);
	void getRays(BaseBuffer * dst);
	
	void update();
	
protected:
    
private:
	void formRays();
	void rayTraverse();
	
private:
	CUDABuffer * m_rays;
	CudaLinearBvh * m_bvh;
    
	unsigned m_numRays, m_rayDim;
	float m_alpha;
};
#endif        //  #ifndef RAY_TEST_H

