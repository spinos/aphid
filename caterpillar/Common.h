#pragma once
#include <AllMath.h>
#include <btBulletCollisionCommon.h>
#include <btBulletDynamicsCommon.h>
namespace caterpillar {
class Common
{
public:

static btTransform CopyFromMatrix44F(const Matrix44F & tm)
{
    const btMatrix3x3 rot(tm.M(0, 0), tm.M(1, 0), tm.M(2, 0), 
                    tm.M(0, 1), tm.M(1, 1), tm.M(2, 1),
                    tm.M(0, 2), tm.M(1, 2), tm.M(2, 2));
    btTransform r;
    r.setBasis(rot);
    const btVector3 pos(tm.M(3, 0), tm.M(3, 1), tm.M(3, 2));
    r.setOrigin(pos);
    return r;
}

static Matrix44F CopyFromBtTransform(const btTransform & tm)
{
	const btMatrix3x3 rot = tm.getBasis();
	Matrix44F r;
	*r.m(0, 0) = rot[0][0];
	*r.m(0, 1) = rot[1][0];
	*r.m(0, 2) = rot[2][0];
	*r.m(1, 0) = rot[0][1];
	*r.m(1, 1) = rot[1][1];
	*r.m(1, 2) = rot[2][1];
	*r.m(2, 0) = rot[0][2];
	*r.m(2, 1) = rot[1][2];
	*r.m(2, 2) = rot[2][2];
	const btVector3 pos = tm.getOrigin();
	*r.m(3, 0) = pos[0];
	*r.m(3, 1) = pos[1];
	*r.m(3, 2) = pos[2];
	return r;
}

};

}
