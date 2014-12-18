#ifndef BOXPROGRAM_H
#define BOXPROGRAM_H

#include <CUDAProgram.h>
#include <AllMath.h>
class CUDABuffer;
class BoxProgram : public CUDAProgram {
public:
	BoxProgram();
	virtual ~BoxProgram();
	
	void createCvs(unsigned numCvs);
    void createIndices(unsigned numIndices, unsigned * src);
	void createAabb(unsigned n);
	void getAabb(Vector3F * dst, unsigned nbox);
	
	virtual void run(Vector3F * pos, unsigned numTriangle, unsigned numVertices);
private:
    CUDABuffer * m_cvs;
    CUDABuffer * m_indices;
    CUDABuffer * m_aabb;
};
#endif        //  #ifndef BOXPROGRAM_H

