/*
 *  FitDeformer.cpp
 *  fit
 *
 *  Created by jian zhang on 4/21/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "FitDeformer.h"

FitDeformer::FitDeformer() {}
FitDeformer::~FitDeformer() {}

void FitDeformer::setTarget(KdTree * tree)
{
	m_tree = tree;
}
	
void FitDeformer::fit()
{
	IntersectionContext ctx;
	ctx.setComponentFilterType(PrimitiveFilter::TVertex);
	
	for(int i = 0; i < (int)m_numVertices; i++) {
		Vector3F pp = m_deformedV[i];
		Vector3F pn = m_mesh->getNormals()[i];
		ctx.reset();
		ctx.setNormalReference(pn);
		m_tree->closestPoint(pp, &ctx);
		if(ctx.m_success) m_deformedV[i] = ctx.m_closest;
	}
}
