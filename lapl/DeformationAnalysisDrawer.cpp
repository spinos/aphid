/*
 *  DeformationAnalysisDrawer.cpp
 *  lapl
 *
 *  Created by jian zhang on 3/24/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef WIN32
#include <gExtension.h>
#else
#include <gl_heads.h>
#endif
#include "DeformationAnalysisDrawer.h"

DeformationAnalysisDrawer::DeformationAnalysisDrawer() {}
DeformationAnalysisDrawer::~DeformationAnalysisDrawer() {}

void DeformationAnalysisDrawer::visualize(DeformationAnalysis * analysis)
{
	setWired(1);
	setColor(0.f, 1.f, 0.7f);
	drawMesh(analysis->getMeshA());
	setColor(0.f, .7f, 1.f);
	drawMesh(analysis->getMeshB());
	const unsigned nv = analysis->numVertices();
	Vector3F vi, dc;
	
	beginLine();
	for(unsigned i = 0; i < nv; i++) {
		vi = analysis->restP(i);
		dc = analysis->differential(i);
		setColor(1.f, 0.f, 0.f);
		glVertex3f(vi.x - dc.x, vi.y - dc.y, vi.z - dc.z);
		glVertex3f(vi.x, vi.y, vi.z);
		dc = analysis->transformedDifferential(i);
		setColor(1.f, 1.f, 0.f);
		glVertex3f(vi.x - dc.x, vi.y - dc.y, vi.z - dc.z);
		glVertex3f(vi.x, vi.y, vi.z);
	}
	end();
}