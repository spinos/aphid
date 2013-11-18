#pragma once
#include <SkeletonPose.h>

class PoseDelta : public SkeletonPose {
public:
    PoseDelta();
    virtual ~PoseDelta();
    virtual void setDegreeOfFreedom(const std::vector<Float3> & dof);
    
    Vector3F * delta() const;

protected:
    
private:
	Vector3F * _delta;
};
