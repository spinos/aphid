/*
 *  SkeletonPose.h
 *  aphid
 *
 *  Created by jian zhang on 10/23/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <AllMath.h>
#include <NamedEntity.h>
class SkeletonJoint;
class SkeletonPose : public NamedEntity {
public:
	SkeletonPose();
	virtual ~SkeletonPose();
	
	void setNumJoints(unsigned x);
	void setDegreeOfFreedom(const std::vector<Float3> & dof);
	void setValues(const std::vector<Float3> & dof, const std::vector<Vector3F> & angles);
	void recoverValues(const std::vector<SkeletonJoint *> & joints);
	unsigned dof() const;
protected:
	
private:
	void cleanup();
	int * m_jointStart;
	float * m_angles;
	unsigned m_numJoints, m_dof;
};