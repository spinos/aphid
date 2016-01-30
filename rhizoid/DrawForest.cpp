/*
 *  DrawForest.cpp
 *  proxyPaint
 *
 *  Created by jian zhang on 1/30/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawForest.h"
#include <gl_heads.h>

DrawForest::DrawForest() 
{ m_defBox = BoundingBox(-1.f, -1.f, -1.f, 1.f, 1.f, 1.f); }

DrawForest::~DrawForest() {}

void DrawForest::drawGround() 
{
	glPushAttrib(GL_CURRENT_BIT);
	glColor3f(.57f, .37f, 0.f);
	
	glBegin(GL_TRIANGLES);
	SelectionContext * active = activeGround();
	std::map<Geometry *, sdb::Sequence<unsigned> * >::iterator it = active->geometryBegin();
	for(; it != active->geometryEnd(); ++it) {
		drawFaces(it->first, it->second);
	}
	glEnd();
	glPopAttrib();
}

void DrawForest::drawFaces(Geometry * geo, sdb::Sequence<unsigned> * components)
{
	ATriangleMesh * mesh = static_cast<ATriangleMesh *>(geo);
	Vector3F *p = mesh->points();
	components->begin();
	while(!components->end()) {
		unsigned * tri = mesh->triangleIndices(components->key() );
		glVertex3fv((GLfloat *)&p[tri[0]]);
		glVertex3fv((GLfloat *)&p[tri[1]]);
		glVertex3fv((GLfloat *)&p[tri[2]]);
		components->next();
	}
}

BoundingBox * DrawForest::defBoxP()
{ return &m_defBox; }

const BoundingBox & DrawForest::defBox() const
{ return m_defBox; }

void DrawForest::draw_solid_box() const
{
	Vector3F minb = m_defBox.getMin();
	Vector3F maxb = m_defBox.getMax();
	
    glBegin(GL_QUADS);
	glNormal3f(0.f, 0.f, -1.f);
	glVertex3f(minb.x, minb.y, minb.z);
	glVertex3f(minb.x, maxb.y, minb.z);
	glVertex3f(maxb.x, maxb.y, minb.z);
	glVertex3f(maxb.x, minb.y, minb.z);
	
	glNormal3f(0.f, 0.f, 1.f);
	glVertex3f(minb.x, minb.y, maxb.z);
	glVertex3f(maxb.x, minb.y, maxb.z);
	glVertex3f(maxb.x, maxb.y, maxb.z);
	glVertex3f(minb.x, maxb.y, maxb.z);
	
	glNormal3f(-1.f, 0.f, 0.f);
	glVertex3f(minb.x, minb.y, minb.z);
	glVertex3f(minb.x, minb.y, maxb.z);
	glVertex3f(minb.x, maxb.y, maxb.z);
	glVertex3f(minb.x, maxb.y, minb.z);
	
	glNormal3f(1.f, 0.f, 0.f);
	glVertex3f(maxb.x, minb.y, minb.z);
	glVertex3f(maxb.x, maxb.y, minb.z);
	glVertex3f(maxb.x, maxb.y, maxb.z);
	glVertex3f(maxb.x, minb.y, maxb.z);
	
	glNormal3f(0.f, -1.f, 0.f);
	glVertex3f(minb.x, minb.y, minb.z);
	glVertex3f(maxb.x, minb.y, minb.z);
	glVertex3f(maxb.x, minb.y, maxb.z);
	glVertex3f(minb.x, minb.y, maxb.z);
	
	glNormal3f(0.f, 1.f, 0.f);
	glVertex3f(minb.x, maxb.y, minb.z);
	glVertex3f(minb.x, maxb.y, maxb.z);
	glVertex3f(maxb.x, maxb.y, maxb.z);
	glVertex3f(maxb.x, maxb.y, minb.z);
	glEnd();
}

void DrawForest::draw_a_box() const
{
	Vector3F minb = m_defBox.getMin();
	Vector3F maxb = m_defBox.getMax();
	
	glBegin( GL_LINES );
	    glVertex3f(minb.x, minb.y, minb.z);
		glVertex3f(maxb.x, minb.y, minb.z);
		glVertex3f(minb.x, maxb.y, minb.z);
		glVertex3f(maxb.x, maxb.y, minb.z);
		glVertex3f(minb.x, minb.y, maxb.z);
		glVertex3f(maxb.x, minb.y, maxb.z);
		glVertex3f(minb.x, maxb.y, maxb.z);
		glVertex3f(maxb.x, maxb.y, maxb.z);
		
		glVertex3f(minb.x, minb.y, minb.z);
		glVertex3f(minb.x, maxb.y, minb.z);
		glVertex3f(maxb.x, minb.y, minb.z);
		glVertex3f(maxb.x, maxb.y, minb.z);
		glVertex3f(minb.x, minb.y, maxb.z);
		glVertex3f(minb.x, maxb.y, maxb.z);
		glVertex3f(maxb.x, minb.y, maxb.z);
		glVertex3f(maxb.x, maxb.y, maxb.z);
		
		glVertex3f(minb.x, minb.y, minb.z);
		glVertex3f(minb.x, minb.y, maxb.z);
		glVertex3f(maxb.x, minb.y, minb.z);
		glVertex3f(maxb.x, minb.y, maxb.z);
		glVertex3f(minb.x, maxb.y, minb.z);
		glVertex3f(minb.x, maxb.y, maxb.z);
		glVertex3f(maxb.x, maxb.y, minb.z);
		glVertex3f(maxb.x, maxb.y, maxb.z);
		
	glEnd();
}

void DrawForest::draw_coordsys() const
{
	Vector3F minb = m_defBox.getMin();
	Vector3F maxb = m_defBox.getMax();
	
	glBegin( GL_LINES );
	glColor3f(1.f, 0.f, 0.f);
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(maxb.x, 0.f, 0.f); 
	glColor3f(0.f, 1.f, 0.f);					
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(0.f, maxb.y, 0.f); 
	glColor3f(0.f, 0.f, 1.f);					
			glVertex3f( 0.f, 0.f, 0.f );
			glVertex3f(0.f, 0.f, maxb.z);		
	glEnd();
}

int DrawForest::activePlantId() const
{ return 0; }

float DrawForest::plantSize(int idx) const
{ return m_defBox.distance(0); }
