#include "PoseDelta.h"
#include <SkeletonPose.h>
PoseDelta::PoseDelta() 
{
	_delta = 0;
}

PoseDelta::~PoseDelta() 
{
	if(_delta) delete[] _delta;
}

void PoseDelta::setDegreeOfFreedom(const std::vector<Float3> & dof)
{
    SkeletonPose::setDegreeOfFreedom(dof);
    _delta = new Vector3F[degreeOfFreedom()];
}

Vector3F * PoseDelta::delta() const
{
    return _delta;
}

