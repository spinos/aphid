#ifndef MUSCLE_H
#define MUSCLE_H

#include "MuscleFascicle.h"
#include "BulletSoftBody/btSoftBody.h"
class Muscle
{
public:
    Muscle();
    virtual ~Muscle();
    void addFacicle(const MuscleFascicle &fascicle);
    
    void create(btSoftBodyWorldInfo& worldInfo);
    int numVertices() const;
    
    void addAnchor(btRigidBody* target, int fascicle, int end);
    btSoftBody* getSoftBody();
    
    int vertexIdx(int fascicle, int end) const;
    int fascicleStart(int fascicle) const;
    int fascicleEnd(int fascicle) const;
private:
    std::vector<MuscleFascicle> m_fascicles;
    btSoftBody* m_dynBody;
};
#endif        //  #ifndef MUSCLE_H

