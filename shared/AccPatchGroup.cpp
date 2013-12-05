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
#include <IntersectionContext.h>
#include <PointInsidePolygonTest.h>
#include <BiLinearInterpolate.h>
#include <InverseBilinearInterpolate.h>

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

void AccPatchGroup::recursiveBezierClosestPoint1(IntersectionContext * ctx, int level, unsigned current) const
{
    BezierPatch* patch = m_activeHirarchy->patch(current);
    BoundingBox controlbox = patch->controlBBox();
	if(!controlbox.isPointAround(ctx->m_originP, ctx->m_minHitDistance)) return;
	PointInsidePolygonTest &pl = *m_activeHirarchy->plane(current);
	Vector3F px;
	char inside = 1;
	const float d = pl.distanceTo(ctx->m_originP, px, inside);
	
	if(m_activeHirarchy->endLevel(level) || !inside) {
		if(d > ctx->m_minHitDistance) return;
		ctx->m_minHitDistance = d;
		ctx->m_componentIdx = ctx->m_curComponentIdx;
		ctx->m_closestP = px;
		ctx->m_hitP = px;
		ctx->m_patchUV = m_activeHirarchy->restoreUV(current, px);
		return;
	}
	
	ctx->m_elementHitDistance = d;
	
	level++;
	
	const unsigned cs = m_activeHirarchy->childStart(current);
	recursiveBezierClosestPoint1(ctx, level, cs);
	recursiveBezierClosestPoint1(ctx, level, cs + 1);
	recursiveBezierClosestPoint1(ctx, level, cs + 2);
	recursiveBezierClosestPoint1(ctx, level, cs + 3);
}

void AccPatchGroup::setActiveHirarchy(unsigned idx)
{
	m_activeHirarchy = &m_hirarchy[idx];
	if(m_hirarchy[idx].isEmpty()) m_hirarchy[idx].create(&m_bezier[idx]);
}