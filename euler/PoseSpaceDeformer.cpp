/*
 *  PoseSpaceDeformer.cpp
 *  eulerRot
 *
 *  Created by jian zhang on 11/6/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "PoseSpaceDeformer.h"

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

void PoseSpaceDeformer::updatePose(unsigned idx)
{
	PoseDelta * pose = findPose(idx);
	if(!pose) return;
	
	const unsigned n = numRestP();
	for(unsigned i = 0; i < n; i++) {
// compute pose->_delta[i];
		m_delta[i] = pose->_delta[i];
	}
}

PoseSpaceDeformer::PoseDelta * PoseSpaceDeformer::findPose(unsigned idx)
{
	PoseDelta * pose = 0;
	std::vector<PoseDelta *>::iterator it = m_poses.begin();
	for(; it != m_poses.end(); ++it) {
		if((*it)->_poseIdx == idx)
			pose = *it;
	}
	return pose;
}