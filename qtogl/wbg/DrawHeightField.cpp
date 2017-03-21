/*
 *  DrawHeightField.cpp
 *  
 *
 *  Created by jian zhang on 3/23/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawHeightField.h"
#include <img/HeightField.h>
#include <math/Ray.h>
#include <gl_heads.h>

using namespace aphid;

DrawHeightField::DrawHeightField() :
m_numVertices(0),
m_curFieldInd(-1),
m_planeHeight(10000.f)
{}

DrawHeightField::~DrawHeightField()
{}

void DrawHeightField::bufferValue(const img::HeightField & fld)
{
	const int & l = fld.numLevels();
	int useLevel = l - 1;
	
	for(int i=0;i<l;++i) {
		const Array3<float> & sig = fld.levelSignal(i);
		if(sig.numCols() <= 64
			|| sig.numRows() <= 64) {
			useLevel = i;
			break;
		}
	}
	
	const Array2<float> * slice = fld.levelSignal(useLevel).rank(0);
	const int & m = slice->numRows();
	const int & n = slice->numCols();
	
	const float scaling = fld.range().x / (float)n;
	
	m_numVertices = m * n * 6;
	m_pos.reset(new float[m_numVertices * 3]);
	m_col.reset(new float[m_numVertices * 3]);
	
	float cbuf[3];
	float pbuf[3] = {0.f, m_planeHeight, 0.f};
	static const float offset[6][2] = {{-.33f, -.33f},
	{-.33f,  .33f},
	{ .33f, -.33f},
	{ .33f,  .33f},
	{ .33f, -.33f},
	{-.33f,  .33f}};
	Float2 cen;
	
	int acc = 0;
	for(int j=0;j<n;++j) {
		const float * colj = slice->column(j);
		cen.x = .5f + (float)j;
		
		for(int i=0;i<m;++i) {
			cen.y = .5f + (float)i;
			
			const float & val = colj[i];
			cbuf[0] = cbuf[1] = cbuf[2] = val;
				
			for(int p=0;p<6;++p) {
				m_col[acc * 3] = m_col[acc * 3+1] = m_col[acc * 3+2] = val;
				
				pbuf[0] = (cen.x + offset[p][0]) * scaling;
				pbuf[2] = (cen.y + offset[p][1]) * scaling;
				
				m_pos[acc * 3] = pbuf[0];
				m_pos[acc * 3+1] = pbuf[1];
				m_pos[acc * 3+2] = pbuf[2];
				
				acc++;
			}
		}
	}
	
}

void DrawHeightField::drawBound(const img::HeightField & fld) const
{
	const float & gh = m_planeHeight;
	glPushMatrix();
	float transbuf[16];
	const Matrix44F & fldt = fld.transformMatrix();
	fldt.glMatrix(transbuf);
	glMultMatrixf((const GLfloat*)transbuf);
	
	const Float2 & rng = fld.range();
	glBegin(GL_LINES);
	glColor3f(1.f, 0.f, 0.f);
	glVertex3f(0.f, gh, 0.f);
	glVertex3f(rng.x, gh, 0.f);
	
	glColor3f(0.f, 1.f, 0.f);
	glVertex3f(0.f, gh, 0.f);
	glVertex3f(0.f, gh, rng.y);
	
	glColor3f(.43f, .53f, .47f);
	glVertex3f(0.f, gh, rng.y);
	glVertex3f(rng.x, gh, rng.y);
	
	glVertex3f(rng.x, gh, 0.f);
	glVertex3f(rng.x, gh, rng.y);
	glEnd();
	
	glPopMatrix();
}

void DrawHeightField::drawValue(const img::HeightField & fld) const
{	
	glPushMatrix();
	float transbuf[16];
	const Matrix44F & fldt = fld.transformMatrix();
	fldt.glMatrix(transbuf);
	glMultMatrixf((const GLfloat*)transbuf);
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	
	glColorPointer(3, GL_FLOAT, 0, (const GLfloat*)m_col.get() );
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)m_pos.get() );

	glDrawArrays(GL_TRIANGLES, 0, m_numVertices);
	
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glPopMatrix();
}

void DrawHeightField::setCurFieldInd(int x)
{ m_curFieldInd = x; }

const int & DrawHeightField::curFieldInd() const
{ return m_curFieldInd; }

const float & DrawHeightField::planeHeight() const
{ return m_planeHeight; }
