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
	m_subject = 0;
	m_rotateAxis = AY;
	m_mode = ToolContext::MoveTransform;
	m_started = 0;
	setEntityType(TTransformManipulator);
}

TransformManipulator::~TransformManipulator() 
{
}

void TransformManipulator::attachTo(BaseTransform * subject)
{
	m_subject = subject;
	setTranslation(subject->translation());
	setRotationAngles(subject->rotationAngles());
	setParent(subject->parent());
}

void TransformManipulator::reattach()
{
	attachTo(m_subject);
}

void TransformManipulator::start(const Ray * r)
{
	const Vector3F worldP = worldSpace().getTranslation();
	Matrix44F ps;
	parentSpace(ps);
	Plane pl(ps.transformAsNormal(hitPlaneNormal()), worldP);
	Vector3F hit, d;
	float t;
	if(pl.rayIntersect(*r, hit, t, 1)) {
		d = hit - worldP;
		if(d.length() > 10.f) return;
		
		if(m_mode == ToolContext::RotateTransform) {
			d = d.normal() * 8.f;
			hit = worldP + d;
		}
		m_startPoint = hit;
		m_currentPoint = m_startPoint;
	}
	m_started = 1;
}

void TransformManipulator::perform(const Ray * r)
{
	const Vector3F worldP = worldSpace().getTranslation();
	Matrix44F ps;
	parentSpace(ps);
	Plane pl(ps.transformAsNormal(hitPlaneNormal()), worldP);
	Vector3F hit, d;
    float t;
	if(pl.rayIntersect(*r, hit, t, 1)) {
		d = hit - worldP;
		if(d.length() > 10.f) return;
		
		if(m_mode == ToolContext::RotateTransform) {
			d = d.normal() * 8.f;
			hit = worldP + d;
		}
		
		d = hit - m_currentPoint;
		
		if(m_mode == ToolContext::MoveTransform)
			move(d);
		else
			spin(d);
			
		m_currentPoint = hit;
	}
}

void TransformManipulator::move(const Vector3F & d)
{
	Matrix44F ps;
	parentSpace(ps);
	Matrix44F invps = ps;
	invps.inverse();
	
	Vector3F od = invps.transformAsNormal(d);
		
	m_subject->translate(od);
	setTranslation(m_subject->translation());
}

void TransformManipulator::spin(const Vector3F & d)
{
	Matrix44F ps;
	parentSpace(ps);
	Matrix44F invps = ps;
	invps.inverse();
	
	const Vector3F worldP = ps.transform(translation());
	const Vector3F rotUp = ps.transformAsNormal(hitPlaneNormal());
	
	Vector3F toa = m_currentPoint - worldP;
	Vector3F tob = toa + d;
	
	toa.normalize();
	tob.normalize();
	float ang = toa.angleBetween(tob, toa.cross(rotUp).reversed());
	
	Vector3F angles;
	
	if(m_rotateAxis == AY) angles.set(0.f, ang, 0.f);
	else if(m_rotateAxis == AZ) angles.set(0.f, 0.f, ang);
	else angles.set(ang, 0.f, 0.f);
	
	m_subject->rotate(angles);
	setRotationAngles(m_subject->rotationAngles());
}

void TransformManipulator::stop()
{
	m_started = 0;
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

Vector3F TransformManipulator::rotatePlane(RotateAxis a) const
{
	return m_subject->rotatePlane(a);
}

Vector3F TransformManipulator::rotationBaseAngles() const
{
	return m_subject->rotationBaseAngles();
}

Vector3F TransformManipulator::hitPlaneNormal() const
{
	if(m_mode == ToolContext::MoveTransform)
		return translatePlane(m_rotateAxis);
	return rotatePlane(m_rotateAxis);
}

Vector3F TransformManipulator::startPoint() const
{
	return m_startPoint;
}

Vector3F TransformManipulator::currentPoint() const
{
	return m_currentPoint;
}

bool TransformManipulator::started() const
{
	return m_started;
}

BaseTransform * TransformManipulator::subject() const
{
	return m_subject;
}