/*
 *  MlCalamus.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlCalamus.h"

MlCalamus::MlCalamus() {}
MlCalamus::~MlCalamus() {}

void MlCalamus::bindToFace(unsigned faceIdx, float u, float v)
{
	m_faceIdx = faceIdx;
	m_patchU = u; 
	m_patchV = v;
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