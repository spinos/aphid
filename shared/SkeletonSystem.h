#pragma once

#include <AllMath.h>
#include <Ray.h>
class SkeletonJoint;
class BaseTransform;
class SkeletonPose;
class SkeletonSystem {
public:
    SkeletonSystem();
    virtual ~SkeletonSystem();
    
    void addJoint(SkeletonJoint * j);
    void clear();
    
    unsigned numJoints() const;
    SkeletonJoint * joint(unsigned idx) const;
    
    SkeletonJoint * selectJoint(const Ray & ray) const;
	
	unsigned degreeOfFreedom() const;
	
	void addPose();
	void selectPose(unsigned i);
	void updatePose();
	void recoverPose();
protected:

private:
	void degreeOfFreedom(BaseTransform * j, std::vector<Float3> & dof) const;
	void rotationAngles(BaseTransform * j, std::vector<Vector3F> & angles) const;
private:
    std::vector<SkeletonJoint *> m_joints;
	std::vector<SkeletonPose *> m_poses;
	SkeletonPose * m_activePose;
};


