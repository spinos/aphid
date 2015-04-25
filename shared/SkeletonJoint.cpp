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
{}

SkeletonJoint::~SkeletonJoint() {}

const TypedEntity::Type SkeletonJoint::type() const
{ return TJoint; }

void SkeletonJoint::setJointOrient(const Vector3F & v)
{
	m_jointOrientAngles = v;
}

Vector3F SkeletonJoint::rotationBaseAngles() const
{
	return m_jointOrientAngles;
}

Vector3F SkeletonJoint::jointOrient() const
{
	return m_jointOrientAngles;
}

void SkeletonJoint::align()
{
	if(numChildren() < 1) return;
	Vector3F t = child(0)->worldSpace().getTranslation();
	Matrix44F ps = worldSpace();
	ps.inverse();
	t = ps.transform(t);
	
	float td = t.length();
	if(td < 10e-5) return;
	t.normalize();
	Vector3F txz(t.x , 0.f, t.z);
	txz.normalize();
	
	Vector3F angles;
	if(txz.z < -10e-5 || txz.z > 10e-5) angles.y = Vector3F::XAxis.angleBetween(txz, Vector3F::ZAxis.reversed());
	if(t.y < -10e-5 || t.y > 10e-5) angles.z = txz.angleBetween(t, Vector3F::YAxis);

	child(0)->setTranslation(Vector3F(td, 0.f, 0.f));
	
	setJointOrient(m_jointOrientAngles + angles);
	
	Vector3F cj = ((SkeletonJoint *)child(0))->jointOrient();
	((SkeletonJoint *)child(0))->setJointOrient(cj + angles.reversed());
	
}

float SkeletonJoint::length() const
{
	if(numChildren() < 1) return 0.f;
	
	Vector3F cp = child(0)->translation();
	return cp.x;
}
