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

void FabricDrawer::drawWale(YarnPatch * patch)
{
	setColor(1.f, 0.f, 0.f);
	short nw = 0;
	unsigned v[4];
	patch->waleEdges(nw, v);
	if(nw < 1) return;

	Vector3F * p = m_positions;
	Vector3F q;
	glBegin(GL_LINES);
	
	q = p[v[0]];
	glVertex3f(q.x, q.y, q.z);
	q = p[v[1]];
	glVertex3f(q.x, q.y, q.z);
	
	if(nw > 1) {
		q = p[v[2]];
		glVertex3f(q.x, q.y, q.z);
		q = p[v[3]];
		glVertex3f(q.x, q.y, q.z);
	}
	glEnd();
}