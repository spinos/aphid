/*
 *  BaseTransform.cpp
 *  eulerRot
 *
 *  Created by jian zhang on 10/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseTransform.h"
#include <Plane.h>
BaseTransform::BaseTransform(BaseTransform * parent)
{
    m_translation.setZero();
    m_angles.setZero();
	m_parent = parent;
	m_rotateDOF.x = m_rotateDOF.y = m_rotateDOF.z = 1.f;
}

BaseTransform::~BaseTransform() {}

void BaseTransform::setParent(BaseTransform * parent)
{
	m_parent = parent;
}

BaseTransform * BaseTransform::parent() const
{
	return m_parent;
}

void BaseTransform::translate(const Vector3F & v)
{
	m_translation += v;
}

void BaseTransform::setTranslation(const Vector3F & v)
{
	m_translation = v;
}

Vector3F BaseTransform::translation() const
{
	return m_translation;
}

void BaseTransform::rotate(const Vector3F & v)
{
	Vector3F v1 = m_angles + v * m_rotateDOF;
	setRotationAngles(v1);
}

void BaseTransform::setRotationAngles(const Vector3F & v)
{
	m_angles = v;
}

Vector3F BaseTransform::rotationBaseAngles() const
{
	return Vector3F();
}

Matrix33F BaseTransform::rotation() const
{
	Matrix33F r;
	Vector3F angs = rotationAngles();
	r.rotateEuler(angs.x, angs.y, angs.z);
	angs = rotationBaseAngles();
	Matrix33F b;
	b.rotateEuler(angs.x, angs.y, angs.z);
	r.multiply(b);
	return r;
}

Vector3F BaseTransform::rotationAngles() const
{
	return m_angles;
}

void BaseTransform::addChild(BaseTransform * child)
{
	m_children.push_back(child);
}

unsigned BaseTransform::numChildren() const
{
	return m_children.size();
}

BaseTransform * BaseTransform::child(unsigned idx) const
{
	return m_children[idx];
}

void BaseTransform::parentSpace(Matrix44F & dst) const
{
	if(!parent()) return;
	
	BaseTransform * p = parent();
	while(p) {
		dst.multiply(p->space());
		p = p->parent();
	}
}

Matrix44F BaseTransform::space() const
{
	Matrix44F s;
	s.setTranslation(m_translation);
	s.setRotation(rotation());
	return s;
}

Matrix44F BaseTransform::worldSpace() const
{
	Matrix44F s = space();
	parentSpace(s);
	return s;
}

bool BaseTransform::intersect(const Ray & ray) const
{	
	Matrix44F s = worldSpace();
	s.inverse();
	Vector3F a = ray.m_origin + ray.m_dir * ray.m_tmin;
	a = s.transform(a);
	Vector3F b = ray.m_origin + ray.m_dir * ray.m_tmax;
	b = s.transform(b);
	Ray objR(a, b);
	
	float hit0, hit1;
	return getBBox().intersect(objR, &hit0, &hit1);
}

Vector3F BaseTransform::translatePlane(RotateAxis a) const
{
	if(a == AX) return Vector3F::XAxis;
	if(a == AY) return Vector3F::YAxis;
	return Vector3F::ZAxis;
}

Vector3F BaseTransform::rotatePlane(RotateAxis a) const
{
	Matrix33F base;
	base.rotateEuler(rotationBaseAngles().x, rotationBaseAngles().y, rotationBaseAngles().z);
	if(a == AZ) return base.transform(Vector3F::ZAxis);
	
	Matrix33F r;
	if(a == AY) {
		r.rotateEuler(0.f, 0.f, rotationAngles().z);
		r.multiply(base);
		return r.transform(Vector3F::YAxis);
	}
	
	r = rotation();
	return r.transform(Vector3F::XAxis);
}

void BaseTransform::detachChild(unsigned idx)
{
	BaseTransform * c = m_children[idx];
	Vector3F p = c->worldSpace().getTranslation();
	c->setTranslation(p);
	c->setParent(0);
	m_children.erase(m_children.begin() + idx);
}

void BaseTransform::setRotateDOF(const Float3 & dof)
{
    m_rotateDOF = dof;
}

Float3 BaseTransform::rotateDOF() const
{
    return m_rotateDOF;
}

const TypedEntity::Type BaseTransform::type() const
{ return TTransform; }