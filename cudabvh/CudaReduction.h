#ifndef CUDAREDUCTION_H
#define CUDAREDUCTION_H
#include "bvh_common.h"
class CUDABuffer;
class CudaReduction {
public:
    CudaReduction();
    virtual ~CudaReduction();
    
    void initOnDevice();
    
    void sumF(float & result, float * idata, unsigned m);
	void maxI(int & result, int * idata, unsigned m);
	void maxF(float & result, float * idata, unsigned m);
	void minF(float & result, float * idata, unsigned m);
	void minX(float * result, Aabb * idata, unsigned m);
	void minPntX(float * result, float3 * idata, unsigned m);
	
	void minMaxF(float * result, float * idata, unsigned m);
	
	void minBox(float * result, Aabb * idata, unsigned m);
	void maxBox(float * result, Aabb * idata, unsigned m);
	void minPnt(float * result, float3 * idata, unsigned m);
	void maxPnt(float * result, float3 * idata, unsigned m);
	
protected:

private:
    CUDABuffer * m_obuf;
};
#endif        //  #ifndef CUDAREDUCTION_H

