/*
 *  AccPatchGroup.cpp
 *  aphid
 *
 *  Created by jian zhang on 11/28/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "AccPatchGroup.h"
#include <accPatch.h>
#include <accStencil.h>
AccPatchGroup::AccPatchGroup() : m_bezier(0) 
{
	if(!AccPatch::stencil) {
		AccStencil* sten = new AccStencil();
		AccPatch::stencil = sten;
	}
}

AccPatchGroup::~AccPatchGroup() 
{
	if(m_bezier) delete[] m_bezier;
}

void AccPatchGroup::createAccPatches(unsigned n)
{
	if(m_bezier) delete[] m_bezier;
	m_bezier = new AccPatch[n];
}

AccPatch* AccPatchGroup::beziers() const
{
	return m_bezier;
}
