/*
 *  DrawCircle.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 2/7/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawCircle.h"
#include <gl_heads.h>

CircleCurve DrawCircle::UnitCircleCurve;

DrawCircle::DrawCircle() {}
DrawCircle::~DrawCircle() {}

void DrawCircle::drawCircle(const float * mat) const
{
	glPushMatrix();
    glMultMatrixf(mat);
	glBegin(GL_LINE_STRIP);
	for(unsigned i = 0; i < UnitCircleCurve.numVertices(); i++) {
		glVertex3fv((const float *)&UnitCircleCurve.getCv(i));
	}
	glEnd();
	glPopMatrix();
}