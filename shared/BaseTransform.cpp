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
	m_rotation.rotateEuler(v.x, v.y, v.z);
}

void BaseTransform::setRotation(const Vector3F & v)
{
	m_rotation.setIdentity();
	m_rotation.rotateEuler(v.x, v.y, v.z);
}

Matrix33F BaseTransform::rotation() const
{
	return m_rotation;
}