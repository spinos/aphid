/*
 *  DrawDop.cpp
 *  
 *
 *  Created by jian zhang on 1/11/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawDop.h"
#include <math/AOrientedBox.h>
#include <math/DOP8Builder.h>
#include <gl_heads.h>

namespace aphid {

DrawDop::DrawDop()
{ 
	m_vertexNormals = 0;
	m_vertexPoints = 0;
	m_vertexColors = 0;
	m_numVertices = 0;
}

DrawDop::~DrawDop()
{ clear(); }

void DrawDop::clear()
{
	if(m_vertexNormals) {
		delete[] m_vertexNormals;
	}
	if(m_vertexPoints) {
		delete[] m_vertexPoints;
	}
	if(m_vertexColors) {
		delete[] m_vertexColors;
	}
}

void DrawDop::update8DopPoints(const AOrientedBox & ob,
                                const float * sizing)
{
	DOP8Builder bd;
	bd.build(ob);
	
	clear();
	
	const int & nt = bd.numTriangles();
	m_numVertices = 3 * nt;
	
	const int nf = m_numVertices*3;
	m_vertexNormals = new float[nf];
	m_vertexPoints = new float[nf];
	m_vertexColors = new float[nf];
	
	const Vector3F * bv = bd.vertex();
	const Vector3F * bn = bd.normal();
	const int * ind = bd.triangleIndices();
	
	for(int i=0;i<m_numVertices;++i) {
		const int i3 = i * 3;
		memcpy(&m_vertexNormals[i3], &bn[i], 12);
		memcpy(&m_vertexPoints[i3], &bv[i], 12);
		
	}
	
	if(!sizing) {
	    return;
	}
	
	for(int i=0;i<m_numVertices;++i) {
		const int i3 = i * 3;
		m_vertexPoints[i3] *= sizing[0];
		m_vertexPoints[i3+1] *= sizing[1];
		m_vertexPoints[i3+2] *= sizing[2];
	}
}

void DrawDop::setUniformDopColor(const float * c)
{
	for(int i=0;i<m_numVertices;++i) {
		memcpy(&m_vertexColors[i*3], c, 12);
	}
}

void DrawDop::drawAWireDop() const
{
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)m_vertexPoints);
	
	glDrawArrays(GL_TRIANGLES, 0, m_numVertices);
}
	
void DrawDop::drawASolidDop() const
{
	glColorPointer(3, GL_FLOAT, 0, (const GLfloat*)m_vertexColors);
	glNormalPointer(GL_FLOAT, 0, (const GLfloat*)m_vertexNormals);
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)m_vertexPoints);

	glDrawArrays(GL_TRIANGLES, 0, m_numVertices);
}

}