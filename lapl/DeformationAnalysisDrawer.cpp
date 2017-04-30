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

void DeformationAnalysisDrawer::visualize(DeformationAnalysis * analysis, bool drawCoord)
{
	setWired(1);
	setColor(0.f, 1.f, 0.2f);
	//drawMesh(analysis->getMeshA());
	setColor(0.f, .1f, 1.f);
	drawMesh(analysis->getMeshB());
	
	if(!drawCoord) return;
	const unsigned nv = analysis->numVertices();
	Vector3F vi, dc;
	
	for(unsigned i = 0; i < nv; i++) {
		vi = analysis->restP(i);
		dc = analysis->differential(i);
		Matrix33F orient = analysis->getR(i);
		glPushMatrix();
		glTranslatef(vi.x, vi.y, vi.z);
		coordsys(orient/*, analysis->getS(i)*/);
		glPopMatrix();
		
		beginLine();
		setColor(1.f, 0.f, 1.f);
		glVertex3f(vi.x - dc.x, vi.y - dc.y, vi.z - dc.z);
		glVertex3f(vi.x, vi.y, vi.z);
		dc = analysis->transformedDifferential(i);
		setColor(1.f, 1.f, 0.f);
		glVertex3f(vi.x - dc.x, vi.y - dc.y, vi.z - dc.z);
		glVertex3f(vi.x, vi.y, vi.z);
		end();
	}
}