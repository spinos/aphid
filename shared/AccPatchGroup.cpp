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
	m_numHirarchy = 0;
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
	m_numHirarchy = n;
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
	PointInsidePolygonTest &pl = *m_activeHirarchy->plane(current);
    const BoundingBox controlbox = pl.getBBox();
	if(!controlbox.isPointAround(ctx->m_originP, ctx->m_minHitDistance)) return;
	
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

void AccPatchGroup::recursiveBezierPatch(int level, unsigned current, std::vector<Vector3F> & dst) const
{
	PointInsidePolygonTest &pl = *m_activeHirarchy->plane(current);
	dst.push_back(pl.vertex(0));
	dst.push_back(pl.vertex(1));
	dst.push_back(pl.vertex(1));
	dst.push_back(pl.vertex(2));
	dst.push_back(pl.vertex(2));
	dst.push_back(pl.vertex(3));
	dst.push_back(pl.vertex(3));
	dst.push_back(pl.vertex(0));
	if(m_activeHirarchy->endLevel(level)) {
		return;
	}
	
	level++;
	
	const unsigned cs = m_activeHirarchy->childStart(current);
	recursiveBezierPatch(level, cs, dst);
	recursiveBezierPatch(level, cs + 1, dst);
	recursiveBezierPatch(level, cs + 2, dst);
	recursiveBezierPatch(level, cs + 3, dst);
}

void AccPatchGroup::setActiveHirarchy(unsigned idx)
{
	m_activeHirarchy = &m_hirarchy[idx];
	if(m_hirarchy[idx].isEmpty()) m_hirarchy[idx].create(&m_bezier[idx]);
}

void AccPatchGroup::setRebuildPatchHirarchy()
{
	for(unsigned i = 0; i < m_numHirarchy; i++) m_hirarchy[i].setRebuild();
}
