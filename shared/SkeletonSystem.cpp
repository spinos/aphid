#include "SkeletonSystem.h"
#include <SkeletonJoint.h>
#include <SkeletonPose.h>
SkeletonSystem::SkeletonSystem() 
{
	m_activePose = 0;
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
	
	std::vector<SkeletonPose *>::iterator itp = m_poses.begin();
    for(; itp != m_poses.end(); ++itp) delete (*itp); 
    m_poses.clear();
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
	degreeOfFreedom(m_joints[0], dofs);
	unsigned ndof = 0;
	std::vector<Float3>::iterator it = dofs.begin();
	for(; it != dofs.end(); ++it) {
		if((*it).x > 0.f) ndof++;
		if((*it).y > 0.f) ndof++;
		if((*it).z > 0.f) ndof++;
	}
	return ndof;
}

void SkeletonSystem::degreeOfFreedom(BaseTransform * j, std::vector<Float3> & dof) const
{
	dof.push_back(j->rotateDOF());
	
	for(unsigned i = 0; i < j->numChildren(); i++) degreeOfFreedom(j->child(i), dof);
}

void SkeletonSystem::rotationAngles(BaseTransform * j, std::vector<Vector3F> & angles) const
{
	angles.push_back(j->rotationAngles());
	for(unsigned i = 0; i < j->numChildren(); i++) rotationAngles(j->child(i), angles);
}

void SkeletonSystem::addPose()
{
	SkeletonPose *pose = new SkeletonPose;
	pose->setName("pose", maxPoseIndex() + 1);
	pose->setIndex(maxPoseIndex() + 1);
	pose->setNumJoints(numJoints());
	std::vector<Float3> dofs;
	degreeOfFreedom(m_joints[0], dofs);
	pose->setDegreeOfFreedom(dofs);
	std::vector<Vector3F> angles;
	rotationAngles(m_joints[0], angles);
	pose->setValues(dofs, angles);
	
	m_poses.push_back(pose);
}

void SkeletonSystem::selectPose(unsigned i)
{
	m_activePose = m_poses[i];
}

void SkeletonSystem::selectPose(const std::string & name)
{
	m_activePose = 0;
	std::vector<SkeletonPose *>::const_iterator it = m_poses.begin();
	for(; it != m_poses.end(); ++it) {
		if((*it)->name() == name) m_activePose = *it;
	}
}
	
void SkeletonSystem::updatePose()
{
	if(!m_activePose) return;
	std::vector<Float3> dofs;
	degreeOfFreedom(m_joints[0], dofs);
	std::vector<Vector3F> angles;
	rotationAngles(m_joints[0], angles);
	m_activePose->setValues(dofs, angles);
}

void SkeletonSystem::recoverPose()
{
	if(!m_activePose) return;
	m_activePose->recoverValues(m_joints);
}

unsigned SkeletonSystem::numPoses() const
{
	return m_poses.size();
}

SkeletonPose * SkeletonSystem::pose(unsigned idx) const
{
	return m_poses[idx];
}

unsigned SkeletonSystem::maxPoseIndex() const
{
	unsigned mx = 0;
	std::vector<SkeletonPose *>::const_iterator it = m_poses.begin();
    for(; it != m_poses.end(); ++it) {
		if((*it)->index() > mx) mx = (*it)->index();
	}
	return mx;
}
