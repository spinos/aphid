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
MlCalamus::MlCalamus() 
{
    m_rotX = m_rotY = m_scale = 0.f;
    m_patchCombined = 0;
}

void MlCalamus::bindToFace(unsigned faceIdx, float u, float v)
{
	m_faceIdx = faceIdx;
	setPatchU(u); 
	setPatchV(v);
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
	unsigned iu = m_patchCombined & EUMask;
	return PATCHPARAMMIN * iu;
}

float MlCalamus::patchV() const
{
    unsigned iv = (m_patchCombined & EVMask) >> EVOFFSET;
	return PATCHPARAMMIN * iv;
}

void MlCalamus::setPatchU(float u)
{
    unsigned iu = u * PATCHPRAAMTIME;
    m_patchCombined = iu + (m_patchCombined & EUMask);
    /*
    std::cout<<"  "<<byte_to_binary(iu)<<"\n";
    std::cout<<"  "<<byte_to_binary(EUMask)<<"\n";
    std::cout<<"  "<<byte_to_binary(EVMask)<<"\n";
    std::cout<<"  "<<byte_to_binary(m_patchCombined)<<"\n";
    */
}

void MlCalamus::setPatchV(float v)
{
    unsigned iv = v * PATCHPRAAMTIME;
    /*std::cout<<"  "<<byte_to_binary(iv)<<"\n";
	std::cout<<"  "<<byte_to_binary(iv<<EVOFFSET)<<"\n";
    std::cout<<"  "<<byte_to_binary(m_patchCombined)<<"\n";
    std::cout<<"  "<<byte_to_binary(~EVMask)<<"\n";*/
    m_patchCombined = (iv<<EVOFFSET) | (m_patchCombined & ~EVMask);
    /*std::cout<<"  "<<byte_to_binary(m_patchCombined)<<"\n";*/
    
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
