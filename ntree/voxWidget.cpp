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
	buildTests();
}

void VoxWidget::clientDraw()
{
	drawGrids();
	drawTriangles();
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
	GridDrawer dr;
	dr.drawGrid<CartesianGrid>(&m_engine);
}

void VoxWidget::drawTriangles()
{
	glColor3f(0,.2,.6);
	glBegin(GL_TRIANGLES);
	const std::vector<cvx::Triangle> & tris = m_engine.prims();
	std::vector<cvx::Triangle>::const_iterator it = tris.begin();
	for(;it!=tris.end();++it) {
		glColor3fv((GLfloat *)&it->C(0) );
		glVertex3fv((GLfloat *)it->p(0) );
		glColor3fv((GLfloat *)&it->C(1) );
		glVertex3fv((GLfloat *)it->p(1) );
		glColor3fv((GLfloat *)&it->C(2) );
		glVertex3fv((GLfloat *)it->p(2) );
	}
	glEnd();
}

void VoxWidget::buildTests()
{
	BoundingBox b(0.f, 0.f, 0.f,
					8.f, 8.f, 8.f);
	m_engine.setBounding(b);
	
	Vector3F vp[3];
	vp[0].set(-1.1f, .1f, -.9f);
	vp[1].set(2.f, 1.1f, 6.9f);
	vp[2].set(9.9f, 3.1f, 3.f);
	
	cvx::Triangle tri;
	tri.resetNC();
	tri.setP(vp[0],0);
	tri.setP(vp[1],1);
	tri.setP(vp[2],2);
	
	Vector3F nor = tri.calculateNormal();
	Vector3F vn[3];
	vn[0] = nor + Vector3F(-.05f, 0.f, 0.f);
	vn[1] = nor + Vector3F(-.05f, 0.f, 0.f);
	vn[2] = nor + Vector3F( .05f, 0.f, 0.f);
	
	tri.setN(vn[0], 0);
	tri.setN(vn[1], 1);
	tri.setN(vn[2], 2);
	
	Vector3F vc[3];
	vc[0].set(1.f, 0.f, 0.f);
	vc[1].set(0.5f, 0.1f, 0.f);
	vc[2].set(0.1f, 0.f, .5f);
	tri.setC(vc[0], 0);
	tri.setC(vc[1], 1);
	tri.setC(vc[2], 2);
	
	m_engine.add(tri);
	
	m_engine.build();
	std::cout<<"\n grid n cell "<<m_engine.numCells();
	std::cout.flush();
}
