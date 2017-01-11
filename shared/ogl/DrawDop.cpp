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
}

void DrawDop::update8DopPoints(const AOrientedBox & ob)
{
	DOP8Builder bd;
	bd.build(ob);
	
	clear();
	
	const int & nt = bd.numTriangles();
	m_numVertices = 3 * nt;
	
	m_vertexNormals = new float[m_numVertices*3];
	m_vertexPoints = new float[m_numVertices*3];
	
	const Vector3F * bv = bd.vertex();
	const Vector3F * bn = bd.normal();
	const int * ind = bd.triangleIndices();
	
	for(int i=0;i<m_numVertices;++i) {
		const int i3 = i * 3;
		m_vertexNormals[i3] = bn[i].x;
		m_vertexNormals[i3+1] = bn[i].y;
		m_vertexNormals[i3+2] = bn[i].z;
		
		m_vertexPoints[i3] = bv[i].x;
		m_vertexPoints[i3+1] = bv[i].y;
		m_vertexPoints[i3+2] = bv[i].z;
	}
}

void DrawDop::drawAWireDop() const
{
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)m_vertexPoints);
	
	glDrawArrays(GL_TRIANGLES, 0, m_numVertices);
}
	
void DrawDop::drawASolidDop() const
{
	glNormalPointer(GL_FLOAT, 0, (const GLfloat*)m_vertexNormals);
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)m_vertexPoints);

	glDrawArrays(GL_TRIANGLES, 0, m_numVertices);
}

}