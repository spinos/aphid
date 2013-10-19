/*
 *  TransformManipulator.cpp
 *  aphid
 *
 *  Created by jian zhang on 10/19/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "TransformManipulator.h"
#include <Plane.h>
TransformManipulator::TransformManipulator() 
{
	m_origin = new BaseTransform;
	m_subject = 0;
	m_rotateAxis = AY;
	m_mode = ToolContext::MoveTransform;
}

TransformManipulator::~TransformManipulator() 
{
	delete m_origin;
}

void TransformManipulator::attachTo(BaseTransform * subject)
{
	m_subject = subject;
	setTranslation(subject->translation());
	setRotation(subject->rotation());
	
	m_origin->setTranslation(subject->translation());
	m_origin->setRotation(subject->rotation());
}

void TransformManipulator::start(const Ray * r)
{
	Plane pl(hitPlaneNormal(), translation());
	
	Vector3F hit, d;
	float t;
	if(pl.rayIntersect(*r, hit, t, 1)) {
		d = hit - translation();
		if(d.length() > 10.f) return;
		m_startPoint = hit;
	}
}

void TransformManipulator::perform(const Ray * r)
{
	Plane pl(hitPlaneNormal(), translation());
	Vector3F hit, d;
    float t;
	if(pl.rayIntersect(*r, hit, t, 1)) {
		d = hit - m_startPoint;
		if(d.length() > 10.f) return;
		if(m_mode == ToolContext::MoveTransform)
			move(d);
		else
			spin(d);
			
		m_startPoint = hit;
	}
}

void TransformManipulator::move(const Vector3F & d)
{
	m_subject->translate(d);
	translate(d);
}

void TransformManipulator::spin(const Vector3F & d)
{
	Vector3F toa = m_startPoint - translation();
	Vector3F tob = toa + d;
	toa.normalize();
	tob.normalize();
	float ang = toa.angleBetween(tob, toa.cross(rotatePlaneNormal(m_rotateAxis)).reversed());
	
	Vector3F angles;
	
	if(m_rotateAxis == AY) angles.set(0.f, ang, 0.f);
	else if(m_rotateAxis == AZ) angles.set(0.f, 0.f, ang);
	else angles.set(ang, 0.f, 0.f);
	
	m_subject->rotate(angles);
	rotate(angles);
}

BaseTransform * TransformManipulator::origin() const
{
	return m_origin;
}

void TransformManipulator::detach()
{
	m_subject = 0;
}

bool TransformManipulator::isDetached() const
{
	return m_subject == 0;
}

void TransformManipulator::setToMove()
{
	m_mode = ToolContext::MoveTransform;
}

void TransformManipulator::setToRotate()
{
	m_mode = ToolContext::RotateTransform;
}

ToolContext::InteractMode TransformManipulator::mode() const
{
	return m_mode;
}

void TransformManipulator::setRotateAxis(RotateAxis axis)
{
	m_rotateAxis = axis;
}

TransformManipulator::RotateAxis TransformManipulator::rotateAxis() const
{
	return m_rotateAxis;
}

Vector3F TransformManipulator::rotatePlaneNormal(RotateAxis a) const
{
	if(a == AZ) return Vector3F::ZAxis;
	Matrix33F m;
	Vector3F r;
	m.rotateZ(rotationAngles().z);
	if(a == AY) {
		r = Vector3F::YAxis;
		r = m.transform(r);
		return r;
	}
	r = Vector3F::XAxis;
	r = rotation().transform(r);
	return r;
}

Vector3F TransformManipulator::translatePlaneNormal() const
{
	if(m_rotateAxis == AX) return Vector3F::XAxis;
	if(m_rotateAxis == AY) return Vector3F::YAxis;
	return Vector3F::ZAxis;
}

Vector3F TransformManipulator::hitPlaneNormal() const
{
	if(m_mode == ToolContext::MoveTransform)
		return translatePlaneNormal();
	return rotatePlaneNormal(m_rotateAxis);
}
