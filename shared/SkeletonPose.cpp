/*
 *  SkeletonPose.cpp
 *  aphid
 *
 *  Created by jian zhang on 10/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "SkeletonPose.h"
#include <SkeletonJoint.h>
SkeletonPose::SkeletonPose() 
{
	m_jointStart = 0;
	m_angles = 0;
}

SkeletonPose::~SkeletonPose() {cleanup();}
	
void SkeletonPose::setNumJoints(unsigned x) 
{
	m_numJoints = x;
	m_jointStart = new int[x];
	for(unsigned i = 0; i < x; i++) m_jointStart[i] = -1;
}

void SkeletonPose::setDegreeOfFreedom(const std::vector<Float3> & dof) 
{
	m_dof = 0;
	std::vector<Float3>::const_iterator it = dof.begin();
	unsigned j, i = 0;
	for(; it != dof.end(); ++it) {
		j = m_dof;
		if((*it).x > 0.f) m_dof++;
		if((*it).y > 0.f) m_dof++;
		if((*it).z > 0.f) m_dof++;
		
		if(m_dof > j) m_jointStart[i] = j;
		i++;
	}
	
	m_angles = new float[m_dof];
}

void SkeletonPose::setValues(const std::vector<Float3> & dof, const std::vector<Vector3F> & angles)
{
	std::vector<Float3>::const_iterator it = dof.begin();
	unsigned i = 0, j = 0;
	Vector3F ang;
	for(; it != dof.end(); ++it) {
		ang = angles[j];
		if((*it).x > 0.f) {
			m_angles[i] = ang.x;
			i++;
		}
		if((*it).y > 0.f) {
			m_angles[i] = ang.y;
			i++;
		}
		if((*it).z > 0.f) {
			m_angles[i] = ang.z;
			i++;
		}
		j++;
	}
}

void SkeletonPose::recoverValues(const std::vector<SkeletonJoint *> & joints) const
{
	unsigned i, j, k = 0;
	for(i = 0; i < joints.size(); i++) {
		if(m_jointStart[i] < 0) continue;
			
		j = m_jointStart[i];
		
		Float3 dof = joints[i]->rotateDOF();
		Vector3F ang;
		
		if(dof.x > 0.f) {
			ang.x = m_angles[k];
			k++;
		}
		if(dof.y > 0.f) {
			ang.y = m_angles[k];
			k++;
		}
		if(dof.z > 0.f) {
			ang.z = m_angles[k];
			k++;
		}
		
		joints[i]->setRotationAngles(ang);
	}
}

void SkeletonPose::cleanup()
{
	if(m_jointStart) delete[] m_jointStart;
	if(m_angles) delete[] m_angles;
}

unsigned SkeletonPose::degreeOfFreedom() const
{
	return m_dof;
}
