/*
 *  DrawCircle.cpp
 *  import math
 *  for i in range(0,33):
 *		a = float(i) / 16
 *  	z = math.cos(math.pi * a)
 *  	y = math.sin(math.pi * a)
 *  	x = 0
 *  	print("{0}, {1}, {2},".format( x, y, z))
 *
 *  Created by jian zhang on 2/7/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawCircle.h"
#include <gl_heads.h>

namespace aphid {

DrawCircle::DrawCircle() {}
DrawCircle::~DrawCircle() {}

void DrawCircle::drawCircle(const float * mat) const
{
	glPushMatrix();
    glMultMatrixf(mat);
	drawCircle();
	glPopMatrix();
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
	
/// x
	drawCircle();
	
/// y
	glRotatef(90, 0, 0, 1);
	glColor3f(0,1,0);
	drawCircle();
	
/// z
	glRotatef(90, 0, 1, 0);
	glColor3f(0,0,1);
	drawCircle();
	
	glPopMatrix();
}

void DrawCircle::drawZCircle(const float * mat) const
{
	glPushMatrix();
    glMultMatrixf(mat);
	
	glRotatef(90, 0, 1, 0);
	drawCircle();
	
	glPopMatrix();
}

void DrawCircle::drawZRing(const float * mat) const
{
	glPushMatrix();
    glMultMatrixf(mat);
	
	glRotatef(90, 0, 1, 0);
	drawCircle();
	
	glScalef(1.1, 1.1, 1.1);
	drawCircle();
	
	glPopMatrix();
}

static const int sUnitCircleNumVertices = 33;

static const float sUnitCircleVertices[] = {
0, 0.0, 1.0,
0, 0.195090322016, 0.980785280403,
0, 0.382683432365, 0.923879532511,
0, 0.55557023302, 0.831469612303,
0, 0.707106781187, 0.707106781187,
0, 0.831469612303, 0.55557023302,
0, 0.923879532511, 0.382683432365,
0, 0.980785280403, 0.195090322016,
0, 1.0, 6.12323399574e-17,
0, 0.980785280403, -0.195090322016,
0, 0.923879532511, -0.382683432365,
0, 0.831469612303, -0.55557023302,
0, 0.707106781187, -0.707106781187,
0, 0.55557023302, -0.831469612303,
0, 0.382683432365, -0.923879532511,
0, 0.195090322016, -0.980785280403,
0, 1.22464679915e-16, -1.0,
0, -0.195090322016, -0.980785280403,
0, -0.382683432365, -0.923879532511,
0, -0.55557023302, -0.831469612303,
0, -0.707106781187, -0.707106781187,
0, -0.831469612303, -0.55557023302,
0, -0.923879532511, -0.382683432365,
0, -0.980785280403, -0.195090322016,
0, -1.0, -1.83697019872e-16,
0, -0.980785280403, 0.195090322016,
0, -0.923879532511, 0.382683432365,
0, -0.831469612303, 0.55557023302,
0, -0.707106781187, 0.707106781187,
0, -0.55557023302, 0.831469612303,
0, -0.382683432365, 0.923879532511,
0, -0.195090322016, 0.980785280403,
0, -2.44929359829e-16, 1.0
};

void DrawCircle::drawCircle() const
{
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)sUnitCircleVertices );

	glDrawArrays(GL_LINE_STRIP, 0, sUnitCircleNumVertices );
	
	glDisableClientState(GL_VERTEX_ARRAY);
}

void DrawCircle::drawXRing() const
{
	glPushMatrix();
    
	drawCircle();
	
	glScalef(1.1, 1.1, 1.1);
	drawCircle();
	
	glPopMatrix();
}

void DrawCircle::drawYRing() const
{
	glPushMatrix();
    glRotatef(90, 0, 0, 1);
	
	drawCircle();
	
	glScalef(1.1, 1.1, 1.1);
	drawCircle();
	
	glPopMatrix();
}
	
void DrawCircle::drawZRing() const
{
	glPushMatrix();
    glRotatef(90, 0, 1, 0);
	
	drawCircle();
	
	glScalef(1.1, 1.1, 1.1);
	drawCircle();
	
	glPopMatrix();
}

}