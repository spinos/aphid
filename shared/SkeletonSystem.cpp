#include "SkeletonSystem.h"
#include <SkeletonJoint.h>
#include <SkeletonPose.h>

namespace aphid {

SkeletonSystem::SkeletonSystem() 
{
}

SkeletonSystem::~SkeletonSystem() 
{
    clear();
}

void SkeletonSystem::clear()
{
    std::vector<SkeletonJoint *>::iterator it = m_joints.begin();
    for(; it != m_joints.end(); ++it) delete (*it); 
    m_joints.clear();
}

void SkeletonSystem::addJoint(SkeletonJoint * j)
{
    m_joints.push_back(j);
    j->setIndex(numJoints() - 1);
}

unsigned SkeletonSystem::numJoints() const
{
    return m_joints.size();
}
   
SkeletonJoint * SkeletonSystem::joint(unsigned idx) const
{
    return m_joints[idx];
}

SkeletonJoint * SkeletonSystem::jointByIndex(unsigned idx) const
{
	std::vector<SkeletonJoint *>::const_iterator it = m_joints.begin();
	for(; it != m_joints.end(); ++it) {
		if((*it)->index() == idx) return *it;
	}
	return 0;
}

SkeletonJoint * SkeletonSystem::selectJoint(const Ray & ray) const
{
    std::vector<SkeletonJoint *>::const_iterator it = m_joints.begin();
	for(; it != m_joints.end(); ++it) {
		if((*it)->intersect(ray)) return *it;
	}
	return m_joints[0];
}

unsigned SkeletonSystem::degreeOfFreedom() const
{
	std::vector<Float3> dofs;
	jointDegreeOfFreedom(m_joints[0], dofs);
	unsigned ndof = 0;
	std::vector<Float3>::iterator it = dofs.begin();
	for(; it != dofs.end(); ++it) {
		if((*it).x > 0.f) ndof++;
		if((*it).y > 0.f) ndof++;
		if((*it).z > 0.f) ndof++;
	}
	return ndof;
}

void SkeletonSystem::degreeOfFreedom(std::vector<Float3> & dof) const
{
	jointDegreeOfFreedom(m_joints[0], dof);
}

void SkeletonSystem::rotationAngles(std::vector<Vector3F> & angles) const
{
	jointRotationAngles(m_joints[0], angles); 
}

void SkeletonSystem::jointDegreeOfFreedom(BaseTransform * j, std::vector<Float3> & dof) const
{
	dof.push_back(j->rotateDOF());
	
	for(unsigned i = 0; i < j->numChildren(); i++) jointDegreeOfFreedom(j->child(i), dof);
}

void SkeletonSystem::jointRotationAngles(BaseTransform * j, std::vector<Vector3F> & angles) const
{
	angles.push_back(j->rotationAngles());
	for(unsigned i = 0; i < j->numChildren(); i++) jointRotationAngles(j->child(i), angles);
}

unsigned SkeletonSystem::closestJointIndex(const Vector3F & pt) const
{
	unsigned res = m_joints[0]->index();
	float curD, minD = 10e8;
	std::vector<SkeletonJoint *>::const_iterator it = m_joints.begin();
	for(; it != m_joints.end(); ++it) {
		curD = (*it)->worldSpace().getTranslation().distance2To(pt);
		if(curD < minD) {
			minD = curD;
			res = (*it)->index();
		}
	}
	
	return res;
}

void SkeletonSystem::calculateBindWeights(const Vector3F & pt, VectorN<unsigned> & ids, VectorN<float> & weights) const
{
	Vector3F q = pt;
	q.z = 0.f;
	
	unsigned i = closestJointIndex(q);
	SkeletonJoint * jointA = jointByIndex(i);
	Matrix44F spaceInv = jointA->worldSpace();
	spaceInv.inverse();
	
	Vector3F subP = spaceInv.transform(q);
	
	if(subP.x < 0.f || jointA->numChildren() < 1) {
		if(jointA->parent()) {
			i = jointA->parent()->index();
			jointA = jointByIndex(i);
			spaceInv = jointA->worldSpace();
			spaceInv.inverse();
			subP = spaceInv.transform(q);
		}
	}
	
	float la = jointA->length();
	
	if(subP.x < 0.f) {
		initIdWeight(1, ids, weights);
		*ids.at(0) = jointA->index();
		*weights.at(0) = 1.f;
		return;
	}
	
	float lowGate = la * .45f;
	float highGate = la * .55f;
	
	if(subP.x >= lowGate && subP.x <= highGate) {
		initIdWeight(1, ids, weights);
		*ids.at(0) = jointA->index();
		*weights.at(0) = 1.f;
		return;
	}
	
	SkeletonJoint * jointB;
	float w;
	
	if(subP.x < lowGate) {
		if(!jointA->parent()) {
			initIdWeight(1, ids, weights);
			*ids.at(0) = jointA->index();
			*weights.at(0) = 1.f;
			return;
		}
		
		jointB = jointByIndex(jointA->parent()->index());
		initIdWeight(2, ids, weights);
		*ids.at(0) = jointA->index();
		*ids.at(1) = jointB->index();
		
		w = .5f + .5f * subP.x / lowGate;
		*weights.at(0) = w;
		*weights.at(1) = 1.f - w;
	}
	else {
		jointB = jointByIndex(jointA->child(0)->index());
		initIdWeight(2, ids, weights);
		*ids.at(0) = jointA->index();
		*ids.at(1) = jointB->index();
		
		w = 1.f - .5f * (subP.x - highGate)/(la - highGate);
		*weights.at(0) = w;
		*weights.at(1) = 1.f - w;
	}
}

void SkeletonSystem::initIdWeight(unsigned n, VectorN<unsigned> & ids, VectorN<float> & weights) const
{
	ids.setZero(n);
	weights.setZero(n);
}

void SkeletonSystem::recoverPose(const SkeletonPose * pose)
{
    pose->recoverValues(m_joints);
}

}