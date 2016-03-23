/*
 *  TriWidget.cpp
 *  
 *
 *  Created by jian zhang on 3/23/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include <QtOpenGL>
#include "triWidget.h"
#include <BaseCamera.h>
#include <GeoDrawer.h>
#include <NTreeDrawer.h>

TriWidget::TriWidget(const std::string & filename, QWidget *parent) : 
Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
    m_intersectCtx.m_success = 0;
	
	if(filename.size() > 1) m_container.readTree(filename);
}

TriWidget::~TriWidget()
{}

void TriWidget::clientInit()
{ connect(internalTimer(), SIGNAL(timeout()), this, SLOT(update())); }

void TriWidget::clientDraw()
{
	drawTriangle();
	// drawTree();
	// drawIntersect();
	// drawGrid();
}

void TriWidget::drawTriangle()
{
	if(!m_container.source() ) return;
	
	//getDrawer()->m_wireProfile.apply();
	getDrawer()->m_surfaceProfile.apply();
	getDrawer()->setColor(.8f, .8f, .8f);
	
/*
	float diff[4] = {0.8, 0.8, 0.8, 1.0};
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_TEXTURE_2D);
	glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, diff );
*/
		
	const sdb::VectorArray<cvx::Triangle> * src = m_container.source();
	const int n = src->size();
	glBegin(GL_TRIANGLES);
	int i=0;
	for(;i<n;++i) {
		const cvx::Triangle * t = src->get(i);
		
		glNormal3fv((GLfloat *)&t->N(0) );
		glVertex3fv((GLfloat *)t->p(0) );
		
		glNormal3fv((GLfloat *)&t->N(1) );
		glVertex3fv((GLfloat *)t->p(1) );
		
		glNormal3fv((GLfloat *)&t->N(2) );
		glVertex3fv((GLfloat *)t->p(2) );
	}
	glEnd();
}

void TriWidget::drawTree()
{
	if(!m_container.tree() ) return; 
	
	getDrawer()->setColor(.15f, .25f, .35f);
	getDrawer()->boundingBox(m_container.tree()->getBBox() );
	
	NTreeDrawer dr;
	dr.drawTree<cvx::Triangle>(m_container.tree() );
}
