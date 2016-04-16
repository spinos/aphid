/*
 *  voxWidget.cpp
 *  
 *
 *  Created by jian zhang on 4/9/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <QtGui>
#include <QtOpenGL>
#include "voxWidget.h"
#include <BaseCamera.h>
#include <GeoDrawer.h>
#include "GridDrawer.h"
using namespace aphid;

VoxWidget::VoxWidget(QWidget *parent) : 
Base3DView(parent)
{
	perspCamera()->setFarClipPlane(20000.f);
	perspCamera()->setNearClipPlane(1.f);
	orthoCamera()->setFarClipPlane(20000.f);
	orthoCamera()->setNearClipPlane(1.f);
	usePerspCamera();

}

VoxWidget::~VoxWidget()
{}

void VoxWidget::clientInit()
{
	m_test.build();
}

void VoxWidget::clientDraw()
{
	drawGrids();
	drawTriangles();
	drawFronts();
}

void VoxWidget::clientMouseInput(QMouseEvent *event)
{
	setUpdatesEnabled(false);
	
	setUpdatesEnabled(true);
	update();
}

void VoxWidget::keyPressEvent(QKeyEvent *event)
{
	switch (event->key()) {
		case Qt::Key_F:
			//camera()->frameAll(getFrameBox() );
		    break;
		default:
			break;
	}
	
	Base3DView::keyPressEvent(event);
}

void VoxWidget::drawGrids()
{
	glColor3f(0,.6,.4);
	GeoDrawer * dr = getDrawer();
	for(int i=0; i< 6; ++i)
		dr->boundingBox(m_test.m_engine[i].getBBox() );
}

void VoxWidget::drawTriangles()
{
	getDrawer()->m_markerProfile.apply();
	glColor3f(0,.2,.6);
	glBegin(GL_TRIANGLES);
	const int n = m_test.m_tris.size();
	int i = 0;
	for(;i<n;++i) {
		const cvx::Triangle * t = m_test.m_tris[i];
		glColor3fv((GLfloat *)&t->C(0) );
		glVertex3fv((GLfloat *)t->p(0) );
		glColor3fv((GLfloat *)&t->C(1) );
		glVertex3fv((GLfloat *)t->p(1) );
		glColor3fv((GLfloat *)&t->C(2) );
		glVertex3fv((GLfloat *)t->p(2) );
	}
	glEnd();
}

void VoxWidget::drawFronts()
{
	GeoDrawer * dr = getDrawer();
	//dr->m_surfaceProfile.apply();
    dr->m_wireProfile.apply();
	for(int i=0; i< 6; ++i) {
		dr->setColor(m_test.TestColor[i][0],
						m_test.TestColor[i][1],
						m_test.TestColor[i][2]);
		dr->orientedBox(&m_test.m_engine[i].orientedBBox() );
	}
}

