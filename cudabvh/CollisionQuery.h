#ifndef COLLISIONQUERY_H
#define COLLISIONQUERY_H

class CUDABuffer;

class CollisionQuery {
public:
    CollisionQuery();
    virtual ~CollisionQuery();
    
    virtual void initOnDevice();
    
protected:
    void setNumPrimitives(unsigned n);
    const unsigned numPrimitives() const;

private:
    CUDABuffer * m_overlapBuffer;
    unsigned m_numPrimitives;
};
#endif        //  #ifndef COLLISIONQUERY_H

