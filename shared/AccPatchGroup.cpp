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
#include <BezierPatchHirarchy.h>
AccPatchGroup::AccPatchGroup() : m_bezier(0), m_hirarchy(0)
{
	if(!AccPatch::stencil) {
		AccStencil* sten = new AccStencil();
		AccPatch::stencil = sten;
	}
}

AccPatchGroup::~AccPatchGroup() 
{
	if(m_bezier) delete[] m_bezier;
	if(m_hirarchy) delete[] m_hirarchy;
}

void AccPatchGroup::createAccPatches(unsigned n)
{
	if(m_bezier) delete[] m_bezier;
	m_bezier = new AccPatch[n];
	if(m_hirarchy) delete[] m_hirarchy;
	m_hirarchy = new BezierPatchHirarchy[n];
}

AccPatch* AccPatchGroup::beziers() const
{
	return m_bezier;
}

BezierPatchHirarchy * AccPatchGroup::hirarchies() const
{
	return m_hirarchy;
}

void AccPatchGroup::createBezierHirarchy(unsigned idx)
{
	m_hirarchy[idx].create(&m_bezier[idx]);
}
