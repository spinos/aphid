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
#include <Aphid.h>
#include <MlFeatherCollection.h>
MlFeatherCollection * MlCalamus::FeatherLibrary = 0;

MlCalamus::MlCalamus() 
{
    m_rotX = m_rotY = m_scale = 0.f;
    m_featherId = 0;
}

void MlCalamus::bindToFace(unsigned faceIdx, float u, float v)
{
	m_faceIdx = faceIdx;
	setPatchU(u); 
	setPatchV(v);
}

void MlCalamus::computeFeatherWorldP(const Vector3F & origin, const Matrix33F& space)
{
	feather()->computeWorldP(origin, space, rotateY(), scale());
}

MlFeather * MlCalamus::feather() const
{
	return FeatherLibrary->featherExample(m_featherId);
}

unsigned MlCalamus::faceIdx() const
{
	return m_faceIdx;
}

float MlCalamus::patchU() const
{
	//unsigned iu = m_patchCombined & EUMask;
	//return PATCHPARAMMIN * iu;
	return m_patchU;
}

float MlCalamus::patchV() const
{
    //unsigned iv = (m_patchCombined & EVMask) >> EVOFFSET;
	//return PATCHPARAMMIN * iv;
	return m_patchV;
}

void MlCalamus::setPatchU(float u)
{
	m_patchU = u;
}

void MlCalamus::setFeatherId(unsigned x)
{
	m_featherId = x;
}

void MlCalamus::setPatchV(float v)
{
    m_patchV = v;
}

void MlCalamus::setRotateX(const float& x)
{
	m_rotX = x;
	
}

void MlCalamus::setRotateY(const float& y)
{
	if(y > -.1f)
	    m_rotY = y;
}

void MlCalamus::setScale(const float & x)
{
	m_scale = x / feather()->getLength();
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
	return m_scale * feather()->getLength();
}

unsigned MlCalamus::bufferStart() const
{
	return m_bufStart;
}

void MlCalamus::collideWith(CollisionRegion * skin, const BoundingBox & bbox)
{
	skin->resetCollisionRegionAround(m_faceIdx, bbox);
	feather()->setCollision(skin);
}

