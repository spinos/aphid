#include "CudaDynamicWorld.h"
#include <CudaBroadphase.h>

CudaDynamicWorld::CudaDynamicWorld() 
{
    m_broadphase = new CudaBroadphase;
}

CudaDynamicWorld::~CudaDynamicWorld()
{
    delete m_broadphase;
}


