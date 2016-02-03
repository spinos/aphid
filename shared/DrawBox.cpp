/*
 *  DrawBox.cpp
 *  
 *
 *  Created by jian zhang on 2/4/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawBox.h"
#include <gl_heads.h>
#include "BoundingBox.h"

const float DrawBox::UnitBoxLine[24][3] = {
{-.5f, -.5f, -.5f},
{ .5f, -.5f, -.5f},
{-.5f,  .5f, -.5f},
{ .5f,  .5f, -.5f},
	
{-.5f, -.5f,  .5f},
{ .5f, -.5f,  .5f},
{-.5f,  .5f,  .5f},
{ .5f,  .5f,  .5f},
	
{-.5f, -.5f, -.5f},
{-.5f,  .5f, -.5f},
{ .5f, -.5f, -.5f},
{ .5f,  .5f, -.5f},
	
{-.5f, -.5f,  .5f},
{-.5f,  .5f,  .5f},
{ .5f, -.5f,  .5f},
{ .5f,  .5f,  .5f},
	
{-.5f, -.5f, -.5f},
{-.5f, -.5f,  .5f},
{ .5f, -.5f, -.5f},
{ .5f, -.5f,  .5f},
	
{-.5f,  .5f, -.5f},
{-.5f,  .5f,  .5f},
{ .5f,  .5f, -.5f},
{ .5f,  .5f,  .5f}
};

const float DrawBox::UnitBoxNormal[36][3] = {
{ 0.f, 0.f,-1.f},
{ 0.f, 0.f,-1.f},
{ 0.f, 0.f,-1.f},
{ 0.f, 0.f,-1.f},
{ 0.f, 0.f,-1.f},
{ 0.f, 0.f,-1.f},

{ 0.f, 0.f, 1.f},
{ 0.f, 0.f, 1.f},
{ 0.f, 0.f, 1.f},
{ 0.f, 0.f, 1.f},
{ 0.f, 0.f, 1.f},
{ 0.f, 0.f, 1.f},

{-1.f, 0.f, 0.f},
{-1.f, 0.f, 0.f},
{-1.f, 0.f, 0.f},
{-1.f, 0.f, 0.f},
{-1.f, 0.f, 0.f},
{-1.f, 0.f, 0.f},

{ 1.f, 0.f, 0.f},
{ 1.f, 0.f, 0.f},
{ 1.f, 0.f, 0.f},
{ 1.f, 0.f, 0.f},
{ 1.f, 0.f, 0.f},
{ 1.f, 0.f, 0.f},

{ 0.f,-1.f, 0.f},
{ 0.f,-1.f, 0.f},
{ 0.f,-1.f, 0.f},
{ 0.f,-1.f, 0.f},
{ 0.f,-1.f, 0.f},
{ 0.f,-1.f, 0.f},

{ 0.f, 1.f, 0.f},
{ 0.f, 1.f, 0.f},
{ 0.f, 1.f, 0.f},
{ 0.f, 1.f, 0.f},
{ 0.f, 1.f, 0.f},
{ 0.f, 1.f, 0.f}
};

const float DrawBox::UnitBoxTriangle[36][3] = {
{-.5f, -.5f, -.5f}, // back
{ .5f,  .5f, -.5f},
{ .5f, -.5f, -.5f},
{-.5f, -.5f, -.5f},
{-.5f,  .5f, -.5f},
{ .5f,  .5f, -.5f},
	
{-.5f, -.5f,  .5f}, // front
{ .5f, -.5f,  .5f},
{ .5f,  .5f,  .5f},
{ .5f,  .5f,  .5f},
{-.5f,  .5f,  .5f},
{-.5f, -.5f,  .5f},
	
{-.5f, -.5f, -.5f}, // left
{-.5f, -.5f,  .5f},
{-.5f,  .5f, -.5f},
{-.5f,  .5f, -.5f},
{-.5f, -.5f,  .5f},
{-.5f,  .5f,  .5f},
	
{ .5f, -.5f, -.5f}, // right
{ .5f,  .5f, -.5f},
{ .5f, -.5f,  .5f},
{ .5f, -.5f,  .5f},
{ .5f,  .5f, -.5f},
{ .5f,  .5f,  .5f},
	
{-.5f, -.5f, -.5f}, // bottom
{ .5f, -.5f, -.5f},
{ .5f, -.5f,  .5f},
{ .5f, -.5f,  .5f},
{-.5f, -.5f,  .5f},
{-.5f, -.5f, -.5f},
	
{-.5f,  .5f, -.5f}, // top
{-.5f,  .5f,  .5f},
{ .5f,  .5f, -.5f},
{ .5f,  .5f, -.5f},
{-.5f,  .5f,  .5f},
{ .5f,  .5f,  .5f}
};

DrawBox::DrawBox() {}
DrawBox::~DrawBox() {}

void DrawBox::drawWireBox(const float * center, const float * scale) const 
{
	glPushMatrix();
	glTranslatef(center[0], center[1], center[2]);
	glScalef(scale[0], scale[1], scale[2]);
	glBegin(GL_LINES);
	for(int i=0;i<24;i++) {
        glVertex3fv(&UnitBoxLine[i][0]);
    }
	glEnd();
	glPopMatrix();
}

void DrawBox::drawSolidBox(const float * center, const float * scale) const 
{
	glPushMatrix();
	glTranslatef(center[0], center[1], center[2]);
	glScalef(scale[0], scale[1], scale[2]);
	glBegin(GL_TRIANGLES);
	for(int i=0;i<36;i++) {
		glNormal3fv(&UnitBoxNormal[i][0]);
        glVertex3fv(&UnitBoxTriangle[i][0]);
    }
	glEnd();
	glPopMatrix();
}
	
void DrawBox::drawBoundingBox(const BoundingBox * box) const
{
	float t[3];
	t[0] = (box->m_data[3] + box->m_data[0]) * .5f;
	t[1] = (box->m_data[4] + box->m_data[1]) * .5f;
	t[2] = (box->m_data[5] + box->m_data[2]) * .5f;
	float s[3];
	s[0] = box->m_data[3] - box->m_data[0];
	s[1] = box->m_data[4] - box->m_data[1];
	s[2] = box->m_data[5] - box->m_data[2];
	drawWireBox(t, s);
}
