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
#include <math/BoundingBox.h>

namespace aphid {

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

void DrawBox::drawWireBox(const float * center, const float & scale) const
{
#if 0
	glPushMatrix();
	glTranslatef(center[0], center[1], center[2]);
	glScalef(scale, scale, scale);
#if 0
	glBegin(GL_LINES);
	for(int i=0;i<24;i++) {
        glVertex3fv(&UnitBoxLine[i][0]);
    }
	glEnd();
#else
    glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)UnitBoxLine);

	glDrawArrays(GL_LINES, 0, 24);
	
	glDisableClientState(GL_VERTEX_ARRAY);
#endif
	glPopMatrix();
#endif

	glBegin(GL_LINES);
	for(int i=0;i<24;i++) {
        glVertex3f(center[0] + scale * UnitBoxLine[i][0],
					center[1] + scale * UnitBoxLine[i][1],
					center[2] + scale * UnitBoxLine[i][2]);
    }
	glEnd();
	
}

void DrawBox::drawSolidBox(const float * center, const float & scale) const
{
#if 0
	glPushMatrix();
	glTranslatef(center[0], center[1], center[2]);
	glScalef(scale, scale, scale);
#if 0
	glBegin(GL_TRIANGLES);
	for(int i=0;i<36;i++) {
		glNormal3fv(&UnitBoxNormal[i][0]);
        glVertex3fv(&UnitBoxTriangle[i][0]);
    }
	glEnd();
#else
    glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glNormalPointer(GL_FLOAT, 0, (const GLfloat*)UnitBoxNormal);
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)UnitBoxTriangle);

	glDrawArrays(GL_TRIANGLES, 0, 36);
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
#endif
	glPopMatrix();
#endif

	glBegin(GL_TRIANGLES);
	for(int i=0;i<36;i++) {
		glNormal3fv(&UnitBoxNormal[i][0]);
        glVertex3f(center[0] + scale * UnitBoxTriangle[i][0],
					center[1] + scale * UnitBoxTriangle[i][1],
					center[2] + scale * UnitBoxTriangle[i][2]);
    }
	glEnd();
	
}

void DrawBox::drawWireBox(const float * center, const float * scale) const 
{
#if 0
	glPushMatrix();
	glTranslatef(center[0], center[1], center[2]);
	glScalef(scale[0], scale[1], scale[2]);
#if 0
	glBegin(GL_LINES);
	for(int i=0;i<24;i++) {
        glVertex3fv(&UnitBoxLine[i][0]);
    }
	glEnd();
#else
    glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)UnitBoxLine);

	glDrawArrays(GL_LINES, 0, 24);
	
	glDisableClientState(GL_VERTEX_ARRAY);
#endif
	glPopMatrix();
#endif

	glBegin(GL_LINES);
	for(int i=0;i<24;i++) {
        glVertex3f(center[0] + scale[0] * UnitBoxLine[i][0],
					center[1] + scale[1] * UnitBoxLine[i][1],
					center[2] + scale[2] * UnitBoxLine[i][2]);
    }
	glEnd();
}

void DrawBox::drawSolidBox(const float * center, const float * scale) const 
{
#if 0
	glPushMatrix();
	glTranslatef(center[0], center[1], center[2]);
	glScalef(scale[0], scale[1], scale[2]);
#if 0
	glBegin(GL_TRIANGLES);
	for(int i=0;i<36;i++) {
		glNormal3fv(&UnitBoxNormal[i][0]);
        glVertex3fv(&UnitBoxTriangle[i][0]);
    }
	glEnd();
#else
    glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glNormalPointer(GL_FLOAT, 0, (GLfloat*)UnitBoxNormal);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)UnitBoxTriangle);

	glDrawArrays(GL_TRIANGLES, 0, 36);
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
#endif
	glPopMatrix();
#endif

	glBegin(GL_TRIANGLES);
	for(int i=0;i<36;i++) {
		glNormal3fv(&UnitBoxNormal[i][0]);
        glVertex3f(center[0] + scale[0] * UnitBoxTriangle[i][0],
					center[1] + scale[1] * UnitBoxTriangle[i][1],
					center[2] + scale[2] * UnitBoxTriangle[i][2]);
    }
	glEnd();
	
}
	
void DrawBox::drawBoundingBox(const BoundingBox * box) const
{ drawHLWireBox(&box->m_data[0]); }

void DrawBox::drawWiredBoundingBox(const BoundingBox * box) const
{ drawHLBox(&box->m_data[0]); }

void DrawBox::drawSolidBoundingBox(const BoundingBox * box) const
{ drawHLSolidBox(&box->m_data[0]); }

void DrawBox::drawSolidBoxArray(const float * data,
						const unsigned & count,
						const unsigned & stride) const
{
	unsigned i=0;
	for(;i<count;i+=stride)
	    drawSolidBox((const float *)&data[i*4], data[i*4+3] );
}

void DrawBox::drawWireBoxArray(const float * data,
						const unsigned & count,
						const unsigned & stride) const
{
	unsigned i=0;
	for(;i<count;i+=stride)
	    drawWireBox((const float *)&data[i*4], data[i*4+3] );
}

void DrawBox::setSolidBoxDrawBuffer(const float * center, const float & scale,
						Vector3F * position, Vector3F * normal) const 
{
	for(int i=0;i<36;i++) {
		normal[i].set(UnitBoxNormal[i][0], 
						UnitBoxNormal[i][1], 
						UnitBoxNormal[i][2]);
        position[i].set(UnitBoxTriangle[i][0] * scale + center[0],
						UnitBoxTriangle[i][1] * scale + center[1],
						UnitBoxTriangle[i][2] * scale + center[2]);
    }
	
}

void DrawBox::drawWiredTriangleArray(const float * ps,
						const unsigned & count) const
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)ps);

	glDrawArrays(GL_TRIANGLES, 0, count);
	
	glDisableClientState(GL_VERTEX_ARRAY);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	
}

void DrawBox::drawSolidBoxArray(const float * ps,
						const float * ns,
						const unsigned & count) const
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glNormalPointer(GL_FLOAT, 0, (GLfloat*)ns);
	glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)ps);

	glDrawArrays(GL_TRIANGLES, 0, count);
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
}

const int DrawBox::HLBoxLine[24][3] = {
{0, 1, 2},
{3, 1, 2},
{0, 4, 2},
{3, 4, 2},
	
{0, 1, 5},
{3, 1, 5},
{0, 4, 5},
{3, 4, 5},
	
{0, 1, 2},
{0, 4, 2},
{3, 1, 2},
{3, 4, 2},
	
{0, 1, 5},
{0, 4, 5},
{3, 1, 5},
{3, 4, 5},
	
{0, 1, 2},
{0, 1, 5},
{3, 1, 2},
{3, 1, 5},
	
{0, 4, 2},
{0, 4, 5},
{3, 4, 2},
{3, 4, 5}
};

void DrawBox::drawHLWireBox(const float * v) const
{
	glBegin(GL_LINES);
	for(int i=0;i<24;i++) {
        glVertex3f(v[HLBoxLine[i][0]], 
					v[HLBoxLine[i][1]], 
					v[HLBoxLine[i][2]]);
    }
	glEnd();
}

const int DrawBox::HLBoxTriangle[36][3] = {
{0, 1, 2}, // back
{3, 4, 2},
{3, 1, 2},
{0, 1, 2},
{0, 4, 2},
{3, 4, 2},
	
{0, 1, 5}, // front
{3, 1, 5},
{3, 4, 5},
{3, 4, 5},
{0, 4, 5},
{0, 1, 5},
	
{0, 1, 2}, // left
{0, 1, 5},
{0, 4, 2},
{0, 4, 2},
{0, 1, 5},
{0, 4, 5},
	
{3, 1, 2}, // right
{3, 4, 2},
{3, 1, 5},
{3, 1, 5},
{3, 4, 2},
{3, 4, 5},
	
{0, 1, 2}, // bottom
{3, 1, 2},
{3, 1, 5},
{3, 1, 5},
{0, 1, 5},
{0, 1, 2},
	
{0, 4, 2}, // top
{0, 4, 5},
{3, 4, 2},
{3, 4, 2},
{0, 4, 5},
{3, 4, 5}
};

void DrawBox::drawHLBox(const float * v) const
{
	for(int i=0;i<36;i++)
		glVertex3f(v[HLBoxTriangle[i][0]], 
						v[HLBoxTriangle[i][1]], 
						v[HLBoxTriangle[i][2]]);
}

void DrawBox::drawHLSolidBox(const float * v) const
{
	glBegin(GL_TRIANGLES);
	for(int i=0;i<36;i++) {
		glNormal3f(UnitBoxNormal[i][0], 
						UnitBoxNormal[i][1], 
						UnitBoxNormal[i][2]);
        glVertex3f(v[HLBoxTriangle[i][0]], 
						v[HLBoxTriangle[i][1]], 
						v[HLBoxTriangle[i][2]]);
    }
	glEnd();
}

}