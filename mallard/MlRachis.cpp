/*
 *  MlRachis.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/20/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlRachis.h"

MlRachis::MlRachis() : m_spaces(0), m_angles(0) {}
MlRachis::~MlRachis() 
{
	if(m_spaces) delete[] m_spaces;
	if(m_angles) delete[] m_angles;
}

void MlRachis::create(unsigned x)
{
	if(m_spaces) delete[] m_spaces;
	if(m_angles) delete[] m_angles;
	m_numSpace = x;
	m_spaces = new Matrix33F[x];
	m_angles = new float[x];
}

void MlRachis::computeAngles(float * segL, float fullL)
{
	for(unsigned i = 0; i < m_numSpace; i++) {
		const float fac = (float)i/(float)m_numSpace;
		m_angles[i] = segL[i] / fullL * (1.f - fac) + sqrt(segL[i] / fullL) * fac;
	}
}

void MlRachis::reset()
{
	for(unsigned i = 0; i < m_numSpace; i++) m_spaces[i].setIdentity(); 
}

void MlRachis::update(const float & fullPitch)
{
	reset();
	for(unsigned i = 0; i < m_numSpace; i++) {
		m_spaces[i].rotateY(fullPitch * m_angles[i]);
	}
}

Matrix33F MlRachis::getSpace(short idx) const
{
	return m_spaces[idx];
}