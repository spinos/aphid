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
#include <zEXRImage.h>

BaseDrawer::BaseDrawer () : m_wired(0) 
{
	m_activeColor.set(0.f, .8f, .2f);
	m_inertColor.set(0.1f, 0.6f, 0.1f);
}

BaseDrawer::~BaseDrawer () 
{
}

void BaseDrawer::initializeProfile()
{
	m_markerProfile = GProfile(false, true, false, false, false);
	m_surfaceProfile = GProfile(true, true, false, false, true);
	surfaceMat = new GMaterial(Color4(0.1, 0.1, 0.1, 1.0),Color4(0.8, 0.8, 0.8, 1.0),Color4(0.4, 0.4, 0.3, 1.0),Color4(0.0, 0.0, 0.0, 1.0), 64.f);
	m_surfaceProfile.m_material = surfaceMat;
	m_wireProfile = GProfile(false, true, true, false, false);
	majorLit.activate();
	fillLit.m_LightID = GL_LIGHT1;
	fillLit.m_Position = Float4(0.f, 0.f, -1000.f, 1.f);
	fillLit.activate();
}

void BaseDrawer::setGrey(float g) const
{
    glColor3f(g, g, g);
}

void BaseDrawer::setColor(float r, float g, float b) const
{
	glColor3f(r, g, b);
}

void BaseDrawer::useColor(const Float3 & c) const
{
	glColor3fv((float *)&c);
}

void BaseDrawer::end() const
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

void BaseDrawer::beginPoint(float x) const
{
	glPointSize(x);
	glBegin(GL_POINTS);
}

void BaseDrawer::beginQuad() const
{
	glBegin(GL_QUADS);
}

void BaseDrawer::boundingRectangle(const BoundingRectangle & b) const
{
	glBegin(GL_LINE_LOOP);
	glVertex3f(b.getMin(0), b.getMin(1), 0.f);
	glVertex3f(b.getMax(0), b.getMin(1), 0.f);
	glVertex3f(b.getMax(0), b.getMax(1), 0.f);
	glVertex3f(b.getMin(0), b.getMax(1), 0.f);
	glEnd();
}

void BaseDrawer::boundingBox(const BoundingBox & b) const
{
	Vector3F corner0(b.getMin(0), b.getMin(1), b.getMin(2));
	Vector3F corner1(b.getMax(0), b.getMax(1), b.getMax(2));
	glBegin(GL_LINE_LOOP);
	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner0.x, corner1.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	glVertex3f(corner1.x, corner0.y, corner0.z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner0.x, corner1.y, corner0.z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(corner1.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(corner0.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner0.y, corner0.z);
	glVertex3f(corner1.x, corner0.y, corner1.z);
	glVertex3f(corner0.x, corner0.y, corner1.z);
	glEnd();
	glBegin(GL_LINE_LOOP);
	glVertex3f(corner0.x, corner1.y, corner0.z);
	glVertex3f(corner0.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner1.z);
	glVertex3f(corner1.x, corner1.y, corner0.z);
	glEnd();
}

void BaseDrawer::useSpace(const Matrix44F & s) const
{
	float m[16];
	s.glMatrix(m);
	glMultMatrixf((const GLfloat*)m);
}

void BaseDrawer::useSpace(const Matrix33F & s) const
{
	float m[16];
	s.glMatrix(m);
	glMultMatrixf((const GLfloat*)m);
}

void BaseDrawer::setWired(char var)
{
	if(var) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void BaseDrawer::setCullFace(char var)
{
	if(var) glEnable(GL_CULL_FACE);
	else glDisable(GL_CULL_FACE);
}

void BaseDrawer::useSolid() const
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void BaseDrawer::useWired() const
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void BaseDrawer::colorAsActive()
{
	glColor3f(m_activeColor.x, m_activeColor.y, m_activeColor.z);
}

void BaseDrawer::colorAsInert()
{
	glColor3f(m_inertColor.x, m_inertColor.y, m_inertColor.z);
}

void BaseDrawer::colorAsReference() const
{
	glColor3f(0.1f, 0.23f, 0.34f);
}

void BaseDrawer::vertex(const Vector3F & v) const
{
	glVertex3fv((float *)&v);
}

void BaseDrawer::vertexWithOffset(const Vector3F & v, const Vector3F & o)
{
	glVertex3f(v.x, v.y, v.z);
	glVertex3f(v.x + o.x, v.y + o.y, v.z + o.z);
}

void BaseDrawer::useDepthTest(char on) const
{
	if(on) glEnable(GL_DEPTH_TEST);
	else glDisable(GL_DEPTH_TEST);
}

int BaseDrawer::addTexture()
{
	GLuint tex = 0;
	m_textureNames.push_back(tex);
	return m_textureNames.size() - 1;
}

void BaseDrawer::clearTexture(int idx)
{
	if(idx < 0) return;
	if(idx + 1 > m_textureNames.size()) return;
	GLuint * tex = &m_textureNames[idx];
	if(*tex > 0) glDeleteTextures(1, tex);
}

int BaseDrawer::loadTexture(int idx, ZEXRImage * image)
{
	if(idx < 0) idx = addTexture();
	
	clearTexture(idx);
	
	GLuint * tex = &m_textureNames[idx];
	
	glGenTextures(1, tex);
	
	glBindTexture(GL_TEXTURE_2D, *tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE); 
	const int w = image->_mipmaps[0]->getWidth();
	if(image->m_channelRank == BaseImage::RGB)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F_ARB, w, w, 0, GL_RGB, GL_HALF_FLOAT_ARB, image->_mipmaps[0]->getPixels());
	else
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, w, w, 0, GL_RGBA, GL_HALF_FLOAT_ARB, image->_mipmaps[0]->getPixels());

	glBindTexture( GL_TEXTURE_2D, 0 );
	
	return idx;
}

void BaseDrawer::texture(int idx)
{	
	glDisable(GL_LIGHTING);
	glColor3f(1, 1, 1);
	glEnable(GL_TEXTURE_2D);
	
	bindTexture(idx);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0);
	glVertex3f(0, 0, 0);
	glTexCoord2f(1, 0);
	glVertex3f(1, 0, 0);
	glTexCoord2f(1, 1);
	glVertex3f(1, 1, 0);
	glTexCoord2f(0, 1);
	glVertex3f(0, 1, 0);
	glEnd();
	unbindTexture();
	glDisable(GL_TEXTURE_2D);
}

void BaseDrawer::bindTexture(int idx)
{
	glBindTexture(GL_TEXTURE_2D, m_textureNames[idx]);
}

void BaseDrawer::unbindTexture()
{
	glBindTexture(GL_TEXTURE_2D, 0);
}

void BaseDrawer::quad(Vector3F & a, Vector3F & b, Vector3F & c, Vector3F & d, char filled) const
{
	if(filled) glBegin(GL_QUADS);
	else glBegin(GL_LINE_LOOP);
	vertex(a); vertex(b); vertex(c); vertex(d);
	glEnd();
}
//:~
