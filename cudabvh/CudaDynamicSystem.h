#ifndef CUDADYNAMICSYSTEM_H
#define CUDADYNAMICSYSTEM_H

class CUDABuffer;

class CudaDynamicSystem {
public:
    CudaDynamicSystem();
    virtual ~CudaDynamicSystem();
    
    void setNumPoints(unsigned n);
    const unsigned numPoints() const;
    
    virtual void initOnDevice();
    virtual void update(float dt);
    
protected:
    void * positionOnDevice();	
	void * velocityOnDevice();
	void * forceOnDevice();
	
	CUDABuffer * X();
	CUDABuffer * V();
	CUDABuffer * F();
    
private:
    CUDABuffer * m_X;
	CUDABuffer * m_V;
    CUDABuffer * m_F;
    unsigned m_numPoints;
};
#endif        //  #ifndef CUDADYNAMICSYSTEM_H

