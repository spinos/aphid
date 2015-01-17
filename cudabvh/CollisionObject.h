#ifndef COLLISIONOBJECT_H
#define COLLISIONOBJECT_H


class CudaLinearBvh;

class CollisionObject {
public:
    CollisionObject();
    virtual ~CollisionObject();
    
    virtual void initOnDevice();
    virtual void updateBvh();
    virtual void update();
    
    CudaLinearBvh * bvh();
    
protected:

private:
    CudaLinearBvh * m_bvh;
};
#endif        //  #ifndef COLLISIONOBJECT_H

