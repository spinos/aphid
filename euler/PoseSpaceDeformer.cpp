/*
 *  PoseSpaceDeformer.cpp
 *  eulerRot
 *
 *  Created by jian zhang on 11/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "PoseSpaceDeformer.h"
#include <PoseDelta.h>
#include <SkeletonPose.h>
#include <SkeletonSystem.h>
#include <SkeletonJoint.h>

PoseSpaceDeformer::PoseSpaceDeformer() 
{
	m_delta = 0;
}

PoseSpaceDeformer::~PoseSpaceDeformer() 
{
	if(m_delta) delete[] m_delta;
	
	std::vector<PoseDelta *>::iterator it = m_poses.begin();
	for(; it != m_poses.end(); ++it) delete *it;
	m_poses.clear();
}

void PoseSpaceDeformer::bindToSkeleton(SkeletonSystem * skeleton)
{
	SkeletonSubspaceDeformer::bindToSkeleton(skeleton);
	m_delta = new Vector3F[numRestP()];
}

Vector3F PoseSpaceDeformer::bindP(unsigned idx, unsigned j) const
{
	return SkeletonSubspaceDeformer::bindP(idx, j) + m_delta[bindStart(idx) + j];
}
/*
void PoseSpaceDeformer::addPose(unsigned idx)
{
	PoseDelta * pose = new PoseDelta(idx, numRestP());
	m_poses.push_back(pose);
}

void PoseSpaceDeformer::selectPose(unsigned idx)
{
	PoseDelta * pose = findPose(idx);
	if(!pose) return;
	
	const unsigned n = numRestP();
	for(unsigned i = 0; i < n; i++) m_delta[i] = pose->_delta[i];
}
*/
void PoseSpaceDeformer::updatePose(unsigned idx)
{
	PoseDelta * pose = findPose(idx);
	if(!pose) return;
	
	const unsigned n = numRestP();
	for(unsigned i = 0; i < n; i++) {
// compute pose->_delta[i];
		m_delta[i] = pose->delta()[i];
	}
}

PoseDelta * PoseSpaceDeformer::findPose(unsigned idx)
{
	PoseDelta * pose = 0;
	std::vector<PoseDelta *>::iterator it = m_poses.begin();
	for(; it != m_poses.end(); ++it) {
		if((*it)->index() == idx)
			pose = *it;
	}
	return pose;
}

void PoseSpaceDeformer::addPose()
{
	PoseDelta * pose = new PoseDelta;
	pose->setName("pose", maxPoseIndex() + 1);
	pose->setIndex(maxPoseIndex() + 1);
	pose->setNumJoints(skeleton()->numJoints());
	std::vector<Float3> dofs;
	skeleton()->degreeOfFreedom(dofs);
	pose->setDegreeOfFreedom(dofs);
	std::vector<Vector3F> angles;
	skeleton()->rotationAngles(angles);
	pose->setValues(dofs, angles);
	
	m_poses.push_back(pose);
	m_activePose = pose;
}
/*
void PoseSpaceDeformer::selectPose(unsigned i)
{
	m_activePose = m_poses[i];
}*/

void PoseSpaceDeformer::selectPose(const std::string & name)
{
	m_activePose = 0;
	std::vector<PoseDelta *>::const_iterator it = m_poses.begin();
	for(; it != m_poses.end(); ++it) {
		if((*it)->name() == name) m_activePose = *it;
	}
}
	
void PoseSpaceDeformer::updatePose()
{
	if(!m_activePose) return;
	std::vector<Float3> dofs;
	skeleton()->degreeOfFreedom(dofs);
	std::vector<Vector3F> angles;
	skeleton()->rotationAngles(angles);
	m_activePose->setValues(dofs, angles);
}

void PoseSpaceDeformer::recoverPose()
{
	if(!m_activePose) return;
	skeleton()->recoverPose(m_activePose);
}

void PoseSpaceDeformer::renamePose(const std::string & fromName, const std::string & toName)
{
	selectPose(fromName);
	if(!m_activePose) return;
	m_activePose->setName(toName);
}

unsigned PoseSpaceDeformer::numPoses() const
{
	return m_poses.size();
}

SkeletonPose * PoseSpaceDeformer::pose(unsigned idx) const
{
	return m_poses[idx];
}

SkeletonPose * PoseSpaceDeformer::currentPose() const
{
	return m_activePose;
}

unsigned PoseSpaceDeformer::maxPoseIndex() const
{
	unsigned mx = 0;
	std::vector<PoseDelta *>::const_iterator it = m_poses.begin();
    for(; it != m_poses.end(); ++it) {
		if((*it)->index() > mx) mx = (*it)->index();
	}
	return mx;
}