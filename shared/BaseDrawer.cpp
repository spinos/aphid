/*
 *  BaseDrawer.cpp
 *  qtbullet
 *
 *  Created by jian zhang on 7/17/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#ifdef WIN32
#include <gExtension.h>
#else
#include <gl_heads.h>
#endif

#include "BaseDrawer.h"
#include "Matrix33F.h"
#include <cmath>

void BaseDrawer::setGrey(float g)
{
    glColor3f(g, g, g);
}

void BaseDrawer::setColor(float r, float g, float b)
{
	glColor3f(r, g, b);
}

void BaseDrawer::box(float width, float height, float depth)
{
	glBegin(GL_LINES);
	glColor3f(1.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(width, 0.f, 0.f);
	
	glColor3f(0.f, 1.f, 0.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, height, 0.f);
	
	glColor3f(0.f, 0.f, 1.f);
	glVertex3f(0.f, 0.f, 0.f);
	glVertex3f(0.f, 0.f, depth);
	
	glColor3f(0.23f, 0.23f, 0.24f);
	
	glVertex3f(width, 0.f, 0.f);
	glVertex3f(width, 0.f, depth);
	
	glVertex3f(width, 0.f, depth);
	glVertex3f(0.f, 0.f, depth);
	
	glVertex3f(0.f, height, 0.f);
	glVertex3f(width, height, 0.f);
	
	glVertex3f(width, height, 0.f);
	glVertex3f(width, height, depth);
	
	glVertex3f(width, height, depth);
	glVertex3f(0.f, height, depth);
	
	glVertex3f(0.f, height, depth);
	glVertex3f(0.f, height, 0.f);
	
	glVertex3f(width, 0.f, 0.f);
	glVertex3f(width, height, 0.f);
	
	glVertex3f(width, 0.f, depth);
	glVertex3f(width, height, depth);
	
	glVertex3f(0.f, 0.f, depth);
	glVertex3f(0.f, height, depth);
	glEnd();
}

void BaseDrawer::solidCube(float x, float y, float z, float size)
{
	glBegin(GL_QUADS);
	
// bottom
	glVertex3f(x, y, z);
	glVertex3f(x + size, y, z);
	glVertex3f(x + size, y, z + size);
	glVertex3f(x, y, z + size);

// top
	glVertex3f(x, y+ size, z);
	glVertex3f(x + size, y+ size, z);
	glVertex3f(x + size, y+ size, z + size);
	glVertex3f(x, y+ size, z + size);
	
// back	
	glVertex3f(x, y, z);
	glVertex3f(x, y + size, z);
	glVertex3f(x + size, y + size, z);
	glVertex3f(x + size, y, z);
	
// front	
	glVertex3f(x, y, z + size);
	glVertex3f(x, y + size, z + size);
	glVertex3f(x + size, y + size, z + size);
	glVertex3f(x + size, y, z + size);

// left
	glVertex3f(x, y, z);
	glVertex3f(x, y, z + size);
	glVertex3f(x, y + size, z + size);
	glVertex3f(x, y + size, z);
	
// right
	glVertex3f(x + size, y, z);
	glVertex3f(x + size, y, z + size);
	glVertex3f(x + size, y + size, z + size);
	glVertex3f(x + size, y + size, z);
	glEnd();
}

void BaseDrawer::end()
{
    if(m_wired) glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glEnd();
}

void BaseDrawer::beginSolidTriangle()
{
	glBegin(GL_TRIANGLES);
}

void BaseDrawer::beginWireTriangle()
{
    m_wired = 1;
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBegin(GL_TRIANGLES);
}

void BaseDrawer::beginLine()
{
	glBegin(GL_LINES);
}

void BaseDrawer::beginPoint()
{
	glBegin(GL_POINTS);
}

void BaseDrawer::beginQuad()
{
	glBegin(GL_QUADS);
}

void BaseDrawer::aVertex(float x, float y, float z)
{
	glVertex3f(x, y, z);
}

void BaseDrawer::drawSphere()
{
	const float angleDelta = 3.14159269f / 36.f;
	float a0, a1, b0, b1;
	glBegin(GL_LINES);
	for(int i=0; i<72; i++) {
		float angleMin = angleDelta * i;
		float angleMax = angleMin + angleDelta;
		
		a0 = cos(angleMin);
		b0 = sin(angleMin);
		
		a1 = cos(angleMax);
		b1 = sin(angleMax);
		
		glVertex3f(a0, b0, 0.f);
		glVertex3f(a1, b1, 0.f);
		
		glVertex3f(a0, 0.f, b0);
		glVertex3f(a1, 0.f, b1);
		
		glVertex3f(0.f, a0, b0);
		glVertex3f(0.f, a1, b1);
	}
	glEnd();
}

void BaseDrawer::drawCircleAround(const Vector3F& center)
{
	Vector3F nor(center.x, center.y, center.z);
	Vector3F tangent = nor.perpendicular();
	
	Vector3F v0 = tangent * 0.1f;
	Vector3F p;
	const float delta = 3.14159269f / 9.f;
	
	glBegin(GL_LINES);
	for(int i = 0; i < 18; i++) {
		p = nor + v0;
		glVertex3f(p.x, p.y, p.z);
		
		v0.rotateAroundAxis(nor, delta);
		
		p = nor + v0;
		glVertex3f(p.x, p.y, p.z);
	}
	glEnd();
}

void BaseDrawer::drawMesh(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	glEnableClientState(GL_VERTEX_ARRAY);
	if(!deformer)
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)mesh->getVertices());
	else
		glVertexPointer(3, GL_FLOAT, 0, (GLfloat*)deformer->getDeformedData());

// draw a cube
	glDrawElements(GL_TRIANGLES, mesh->getNumFaceVertices(), GL_UNSIGNED_INT, mesh->getIndices());

// deactivate vertex arrays after drawing
	glDisableClientState(GL_VERTEX_ARRAY);
}

void BaseDrawer::tangentFrame(const BaseMesh * mesh, const BaseDeformer * deformer)
{
	unsigned nv = mesh->getNumVertices();
	Vector3F * v = mesh->getVertices();
	if(deformer)
		v = deformer->getDeformedData();
		
	float m[16];
	for(unsigned i = 0; i < nv; i++) {
		glPushMatrix();
		//glTranslatef(v[i].x, v[i].y, v[i].z);
		Matrix33F orient = mesh->getTangentFrame(i);
    
    m[0] = orient(0, 0); m[1] = orient(0, 1); m[2] = orient(0, 2); m[3] = 0.0;
    m[4] = orient(1, 0); m[5] = orient(1, 1); m[6] = orient(1, 2); m[7] = 0.0;
    m[8] = orient(2, 0); m[9] = orient(2, 1); m[10] = orient(2, 2); m[11] = 0.0;
    m[12] = v[i].x;
	m[13] = v[i].y;
	m[14] = v[i].z; 
	m[15] = 1.0;
    glMultMatrixf((const GLfloat*)m);
		coordsys();
				glPopMatrix();
	}
}

void BaseDrawer::box(const BoundingBox & b)
{
	beginQuad();
	Vector3F corner0(b.min(0), b.min(1), b.min(2));
	Vector3F corner1(b.max(0), b.max(1), b.max(2));

	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	glVertex3f(corner0.x, corner1.y, corner0.z);
	
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	
	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner0.z);
	
	glVertex3f(corner1.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	
	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner0.z);
	
	glVertex3f(corner0.x, corner1.y, corner0.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	end();
}

void BaseDrawer::coordsys()
{
	glBegin( GL_LINES );
	glColor3f(1.f, 0.f, 0.f);
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(1.f, 0.f, 0.f); 
	glColor3f(0.f, 1.f, 0.f);					
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(0.f, 1.f, 0.f); 
	glColor3f(0.f, 0.f, 1.f);					
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(0.f, 0.f, 1.f);		
	glEnd();

}

void BaseDrawer::setWired(char var)
{
	if(var) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

