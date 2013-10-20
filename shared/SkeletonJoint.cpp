/*
 *  SkeletonJoint.cpp
 *  aphid
 *
 *  Created by jian zhang on 10/20/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "SkeletonJoint.h"

SkeletonJoint::SkeletonJoint(BaseTransform * parent) : BaseTransform(parent) 
{
	setEntityType(TJoint);
}

SkeletonJoint::~SkeletonJoint() {}

void SkeletonJoint::setJointOrient(const Vector3F & v)
{
	m_jointOrientAngles = v;
}

Vector3F SkeletonJoint::rotationBaseAngles() const
{
	return m_jointOrientAngles;
}

void SkeletonJoint::align()
{
	if(numChildren() < 1) return;
	Vector3F t = child(0)->translation();
	
	
}