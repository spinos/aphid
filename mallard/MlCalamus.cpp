/*
 *  MlCalamus.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlCalamus.h"
#include "MlFeather.h"
#include <CollisionRegion.h>
MlCalamus::MlCalamus() 
{
	m_patchU = m_patchV = m_rotX = m_rotY = m_scale = 0.f;
}

void MlCalamus::bindToFace(unsigned faceIdx, float u, float v)
{
	m_faceIdx = faceIdx;
	m_patchU = u; 
	m_patchV = v;
}

void MlCalamus::computeFeatherWorldP(const Vector3F & origin, const Matrix33F& space)
{
	m_geo->computeWorldP(origin, space, rotateY(), scale());
}

MlFeather * MlCalamus::feather() const
{
	return m_geo;
}

unsigned MlCalamus::faceIdx() const
{
	return m_faceIdx;
}

float MlCalamus::patchU() const
{
	return m_patchU;
}

float MlCalamus::patchV() const
{
	return m_patchV;
}

void MlCalamus::setFeather(MlFeather * geo)
{
	m_geo = geo;
}

void MlCalamus::setRotateX(const float& x)
{
	m_rotX = x;
	
}

void MlCalamus::setRotateY(const float& y)
{
	m_rotY = y;
}

void MlCalamus::setScale(const float & x)
{
	m_scale = x / m_geo->getLength();
}

void MlCalamus::setBufferStart(unsigned x)
{
	m_bufStart = x;
}

float MlCalamus::rotateX() const
{
	return m_rotX;
}

float MlCalamus::rotateY() const
{
	return m_rotY;
}

float MlCalamus::scale() const
{
	return m_scale;
}

float MlCalamus::realScale() const
{
	return m_scale * m_geo->getLength();
}

unsigned MlCalamus::bufferStart() const
{
	return m_bufStart;
}

void MlCalamus::collideWith(CollisionRegion * skin, const Vector3F & p)
{
	skin->resetCollisionRegionAround(m_faceIdx, p, realScale());
	m_geo->setCollision(skin);
}
