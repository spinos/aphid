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

aphid::CircleCurve DrawCircle::UnitCircleCurve;

DrawCircle::DrawCircle() {}
DrawCircle::~DrawCircle() {}

void DrawCircle::drawCircle(const float * mat) const
{
	glPushMatrix();
    glMultMatrixf(mat);
	drawCircle();
	glPopMatrix();
}

void DrawCircle::drawCircle() const
{
#if 0
	glBegin(GL_LINE_STRIP);
	for(unsigned i = 0; i < UnitCircleCurve.numVertices(); i++) {
		glVertex3fv((const float *)&UnitCircleCurve.getCv(i));
	}
	glEnd();
#else
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)UnitCircleCurve.cvV() );

	glDrawArrays(GL_LINE_STRIP, 0, UnitCircleCurve.numVertices() );
	
	glDisableClientState(GL_VERTEX_ARRAY);
#endif
}

void DrawCircle::draw3Circles(const float * mat) const
{
	glPushMatrix();
    glMultMatrixf(mat);
	
	glColor3f(0,0,1);
	glBegin(GL_LINES);
    glVertex3f(0.f, 0.f, 0.f);
    glVertex3f(0, 0, 1);
	
	glColor3f(0,1,0);
    glVertex3f(0.f, 0.f, 0.f);
    glVertex3f(0, 1, 0);
	
	glColor3f(1,0,0);
    glVertex3f(0.f, 0.f, 0.f);
    glVertex3f(1, 0, 0);
    glEnd();
	
	glColor3f(0,0,1);
	drawCircle();
	
	glRotatef(90, 1, 0, 0);
	glColor3f(0,1,0);
	drawCircle();
	
	glRotatef(-90, 1, 0, 0);
	glRotatef(90, 0, 1, 0);
	
	glColor3f(1,0,0);
	drawCircle();
	
	glPopMatrix();
}
