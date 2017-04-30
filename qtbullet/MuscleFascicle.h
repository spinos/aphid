#ifndef MUSCLEFASCICLE_H
#define MUSCLEFASCICLE_H
#include "btBulletDynamicsCommon.h"
#include <vector>

class MuscleFascicle
{
public:
    MuscleFascicle();
    virtual ~MuscleFascicle();
    void addVertex(const float & x, const float & y, const float & z);
    int numVertices() const;
    btVector3 vertexAt(int idx) const;
    
private:
    std::vector<btVector3> m_vertices;
};
#endif        //  #ifndef MUSCLEFASCICLE_H

