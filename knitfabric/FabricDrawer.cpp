/*
 *  FabricDrawer.cpp
 *  knitfabric
 *
 *  Created by jian zhang on 6/5/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef WIN32
#include <gExtension.h>
#else
#include <gl_heads.h>
#endif
#include "FabricDrawer.h"
#include <YarnPatch.h>
FabricDrawer::FabricDrawer() {}

void FabricDrawer::setPositions(Vector3F * p)
{
	m_positions = p;
}

void FabricDrawer::drawYarn(YarnPatch * patch)
{
	if(!patch->hasTessellation()) return;
	setColor(0.f, .9f, .4f);
	//drawPolygons(patch);
	drawPoints(patch);
}

void FabricDrawer::drawWale(YarnPatch * patch)
{
	setColor(1.f, 0.f, 0.f);
	glDisable(GL_DEPTH_TEST);
	short nw = 0;
	unsigned v[4];
	patch->waleEdges(nw, v);
	if(nw < 1) return;

	Vector3F * p = m_positions;
	Vector3F q[4], c, lft, rgt;
	glBegin(GL_LINES);
	
	q[0] = p[v[0]];
	glVertex3f(q[0].x, q[0].y, q[0].z);
	q[1] = p[v[1]];
	glVertex3f(q[1].x, q[1].y, q[1].z);
	
	if(nw > 1) {
		q[2] = p[v[2]];
		glVertex3f(q[2].x, q[2].y, q[2].z);
		q[3] = p[v[3]];
		glVertex3f(q[3].x, q[3].y, q[3].z);
		
		c = q[0] * 0.35f + q[1] * 0.15f + q[2] * 0.35f + q[3] * 0.15f;
		lft = c - q[1];
		lft *= 0.1f;
		
		rgt = c - q[3];
		rgt *= 0.1f;
		
		vertexWithOffset(q[1], lft);
		vertexWithOffset(q[1], rgt);
		vertexWithOffset(q[3], lft);
		vertexWithOffset(q[3], rgt);
	}
	glEnd();
	glEnable(GL_DEPTH_TEST);
}