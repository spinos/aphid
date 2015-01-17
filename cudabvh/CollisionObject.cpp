#include "CollisionObject.h"
#include "CudaLinearBvh.h"

CollisionObject::CollisionObject() 
{
    m_bvh = new CudaLinearBvh;
    
}

CollisionObject::~CollisionObject() {}

void CollisionObject::initOnDevice() 
{ m_bvh->create(); }

void CollisionObject::updateBvh() {}

CudaLinearBvh * CollisionObject::bvh()
{ return m_bvh; }
