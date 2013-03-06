#ifndef SKIN_H
#define SKIN_H

#include "BulletSoftBody/btSoftBody.h"

class Skin
{
public:
    Skin();
    virtual ~Skin();
    void create(btSoftBodyWorldInfo& worldInfo, const char *filename);
    void addAnchor(btRigidBody* target, int idx);
    void drag(float x, float y, float z);
    btSoftBody* getSoftBody();
private:
    btSoftBody* m_dynBody;
};
#endif        //  #ifndef SKIN_H

