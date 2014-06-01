#pragma once
#include <btBulletCollisionCommon.h>
#include <btBulletDynamicsCommon.h>
namespace caterpillar {
class Common
{
public:

static btTransform CopyFromMatrix44F(const Matrix44F & tm)
{
    const btMatrix3x3 rot(tm.M(0, 0), tm.M(0, 1), tm.M(0, 2), 
                    tm.M(1, 0), tm.M(1, 1), tm.M(1, 2),
                    tm.M(2, 0), tm.M(2, 1), tm.M(2, 2));
    btTransform r;
    r.setBasis(rot);
    const btVector3 pos(tm.M(3, 0), tm.M(3, 1), tm.M(3, 2));
    r.setOrigin(pos);
    return r;
}

};

}
