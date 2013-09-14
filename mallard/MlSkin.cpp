/*
 *  MlSkin.cpp
 *  mallard
 *
 *  Created by jian zhang on 9/14/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "MlSkin.h"
#include <AccPatchMesh.h>
MlSkin::MlSkin() : m_numFeather(0) {}
MlSkin::~MlSkin()
{
	m_calamus.clear();
}

void MlSkin::setBodyMesh(AccPatchMesh * mesh)
{
	m_body = mesh;
}

AccPatchMesh * MlSkin::bodyMesh() const
{
	return m_body;
}

void MlSkin::addCalamus(MlCalamus & ori)
{
	m_calamus.expandBy(1);
	MlCalamus * c = m_calamus.asCalamus();
	*c = ori;
	m_calamus.next();
	m_numFeather++;
}

unsigned MlSkin::numFeathers() const
{
	return m_numFeather;
}

MlCalamus * MlSkin::getCalamus(unsigned idx) const
{
	return m_calamus.asCalamus(idx);
}
