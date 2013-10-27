/*
 *  SkeletonJoint.h
 *  aphid
 *
 *  Created by jian zhang on 10/20/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once

#include <BaseTransform.h>

class SkeletonJoint : public BaseTransform {
public:
	SkeletonJoint(BaseTransform * parent = 0);
	virtual ~SkeletonJoint();
	
	void setJointOrient(const Vector3F & v);
	Vector3F jointOrient() const;
	
	void align();
	
	virtual Vector3F rotationBaseAngles() const;
	
	float length() const;
protected:
	
private:
	Vector3F m_jointOrientAngles;
};