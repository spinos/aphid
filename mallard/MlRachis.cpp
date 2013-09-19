/*
 *  MlRachis.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/20/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlRachis.h"

MlRachis::MlRachis() : m_space(0) {}
MlRachis::~MlRachis() 
{
	if(m_space) delete[] m_space;
}

void MlRachis::create(unsigned x)
{
	if(m_space) delete[] m_space;
	m_numSpace = x;
	m_space = new Matrix33F[x];
}

void MlRachis::reset()
{
	for(unsigned i = 0; i < m_numSpace; i++) m_space[i].setIdentity(); 
}

Matrix33F MlRachis::getSpace(short idx) const
{
	return m_space[idx];
}