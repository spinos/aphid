/*
 *  BaseTransform.cpp
 *  eulerRot
 *
 *  Created by jian zhang on 10/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "BaseTransform.h"

BaseTransform::BaseTransform(BaseTransform * parent)
{
	m_parent = parent;
	setEntityType(TTransform);
}

BaseTransform::~BaseTransform() {}

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
	m_angles += v;
	m_rotation.setIdentity();
	m_rotation.rotateEuler(m_angles.x, m_angles.y, m_angles.z);
}

void BaseTransform::setRotation(const Vector3F & v)
{
	m_angles = v;
	m_rotation.setIdentity();
	m_rotation.rotateEuler(m_angles.x, m_angles.y, m_angles.z);
}

void BaseTransform::setRotation(const Matrix33F & m)
{
	m_rotation = m;
}

Matrix33F BaseTransform::rotation() const
{
	return m_rotation;
}

Vector3F BaseTransform::rotationAngles() const
{
	return m_angles;
}