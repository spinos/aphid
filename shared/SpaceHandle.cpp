/*
 *  SpaceHandle.cpp
 *  masq
 *
 *  Created by jian zhang on 5/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "SpaceHandle.h"
namespace aphid {

SpaceHandle::SpaceHandle() 
{
	m_size = 0.5f;
}

SpaceHandle::~SpaceHandle() {}

void SpaceHandle::keepOriginalSpace()
{
	m_space0 = m_space;
}

void SpaceHandle::spaceMatrix(float m[16]) const
{
	m[0] = m_space.M(0,0); m[1] = m_space.M(0,1); m[2] = m_space.M(0,2); m[3] = 0.0;
    m[4] = m_space.M(1,0); m[5] = m_space.M(1,1); m[6] = m_space.M(1,2); m[7] = 0.0;
    m[8] = m_space.M(2,0); m[9] = m_space.M(2,1); m[10] =m_space.M(2,2); m[11] = 0.0;
    m[12] = m_space.M(3,0); m[13] = m_space.M(3,1); m[14] = m_space.M(3,2) ; m[15] = 1.0;
}

Vector3F SpaceHandle::getCenter() const
{
	return m_space.getTranslation();
}

Vector3F SpaceHandle::displacement() const
{
	return m_space.getTranslation() - m_space0.getTranslation();
}

void SpaceHandle::setSize(float val)
{
	m_size = val;
}

float SpaceHandle::getSize() const
{
	return m_size;
}

}