#pragma once

#include <AllMath.h>
#include <Ray.h>
class SkeletonJoint;
class SkeletonSystem {
public:
    SkeletonSystem();
    virtual ~SkeletonSystem();
    
    void addJoint(SkeletonJoint * j);
    void clear();
    
    unsigned numJoints() const;
    SkeletonJoint * joint(unsigned idx) const;
    
    SkeletonJoint * selectJoint(const Ray & ray) const;
protected:

private:
    std::vector<SkeletonJoint *> m_joints;
};


