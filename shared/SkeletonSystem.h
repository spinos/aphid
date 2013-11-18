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
	SkeletonJoint * jointByIndex(unsigned idx) const;
    
    SkeletonJoint * selectJoint(const Ray & ray) const;
	
	unsigned degreeOfFreedom() const;
	void degreeOfFreedom(std::vector<Float3> & dof) const;
	void rotationAngles(std::vector<Vector3F> & angles) const;
	
	unsigned closestJointIndex(const Vector3F & pt) const;
	void calculateBindWeights(const Vector3F & pt, VectorN<unsigned> & ids, VectorN<float> & weights) const;
	
	void recoverPose(const SkeletonPose * pose);
protected:
	void jointDegreeOfFreedom(BaseTransform * j, std::vector<Float3> & dof) const;
	void jointRotationAngles(BaseTransform * j, std::vector<Vector3F> & angles) const;
	
private:
	void initIdWeight(unsigned n, VectorN<unsigned> & ids, VectorN<float> & weights) const;
private:
    std::vector<SkeletonJoint *> m_joints;
};


