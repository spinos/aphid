#include "CollisionObject.h"
#include "CudaLinearBvh.h"

CollisionObject::CollisionObject() 
{
    m_bvh = new CudaLinearBvh;
    
}

CollisionObject::~CollisionObject() {}

void CollisionObject::initOnDevice() 
{ m_bvh->create(); }

void CollisionObject::update() 
{ m_bvh->update(); }

CudaLinearBvh * CollisionObject::bvh()
{ return m_bvh; }
