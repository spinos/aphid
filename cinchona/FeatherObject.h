/*
 *  feather with mesh, transform
 */

#ifndef FEATHER_OBJECT_H
#define FEATHER_OBJECT_H

#include <math/Matrix44F.h>
class FeatherMesh;

class FeatherObject : public aphid::Matrix44F {
    FeatherMesh * m_mesh;
    
public:
    FeatherObject(FeatherMesh * mesh);
    virtual ~FeatherObject();
    
    const FeatherMesh * mesh() const;
    
protected:
};

#endif
