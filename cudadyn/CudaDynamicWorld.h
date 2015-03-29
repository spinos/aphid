#ifndef CUDADYNAMICWORLD_H
#define CUDADYNAMICWORLD_H

class CudaBroadphase;
class CudaDynamicWorld
{
public:
    CudaDynamicWorld();
    virtual ~CudaDynamicWorld();
    
protected:

private:
    CudaBroadphase * m_broadphase;
};
#endif        //  #ifndef CUDADYNAMICWORLD_H

