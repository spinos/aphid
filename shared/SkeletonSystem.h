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
	
	void addPose();
	void selectPose(unsigned i);
	void selectPose(const std::string & name);
	void updatePose();
	void recoverPose();
	void renamePose(const std::string & fromName, const std::string & toName);
	
	unsigned numPoses() const;
	SkeletonPose * pose(unsigned idx) const;
	SkeletonPose * currentPose() const;
	
	unsigned closestJointIndex(const Vector3F & pt) const;
	void calculateBindWeights(const Vector3F & pt, VectorN<unsigned> & ids, VectorN<float> & weights) const;
protected:

private:
	unsigned maxPoseIndex() const;
	void degreeOfFreedom(BaseTransform * j, std::vector<Float3> & dof) const;
	void rotationAngles(BaseTransform * j, std::vector<Vector3F> & angles) const;
	void initIdWeight(unsigned n, VectorN<unsigned> & ids, VectorN<float> & weights) const;
private:
    std::vector<SkeletonJoint *> m_joints;
	std::vector<SkeletonPose *> m_poses;
	SkeletonPose * m_activePose;
};


