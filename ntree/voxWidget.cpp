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
	//drawGrids();
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
	GridDrawer dr;
	dr.drawGrid<CartesianGrid>(&m_engine);
	dr.drawGrid<CartesianGrid>(&m_engine1);
	dr.drawGrid<CartesianGrid>(&m_engine2);
	dr.drawGrid<CartesianGrid>(&m_engine3);
	dr.drawGrid<CartesianGrid>(&m_engine4);
	dr.drawGrid<CartesianGrid>(&m_engine5);
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

void VoxWidget::drawFronts()
{
	GeoDrawer * dr = getDrawer();
	dr->setColor(.1f, .1f, .1f);
	dr->orientedBox(&m_engine.orientedBBox() );
	dr->setColor(0.f, .1f, .5f);
	dr->orientedBox(&m_engine1.orientedBBox() );
	dr->setColor(.9f, .1f, .1f);
	dr->orientedBox(&m_engine2.orientedBBox() );
	dr->setColor(.9f, .9f, .1f);
	dr->orientedBox(&m_engine3.orientedBBox() );
	dr->setColor(0.f, .3f, .5f);
	dr->orientedBox(&m_engine4.orientedBBox() );
	dr->setColor(0.1f, .3f, 0.f);
	dr->orientedBox(&m_engine5.orientedBBox() );
}

cvx::Triangle VoxWidget::createTriangle(const Vector3F & p0,
								const Vector3F & p1,
								const Vector3F & p2,
								const Vector3F & c0,
								const Vector3F & c1,
								const Vector3F & c2)
{
	cvx::Triangle tri;
	tri.resetNC();
	tri.setP(p0,0);
	tri.setP(p1,1);
	tri.setP(p2,2);
	Vector3F nor = tri.calculateNormal();
	tri.setN(nor, 0);
	tri.setN(nor, 1);
	tri.setN(nor, 2);
	tri.setC(c0, 0);
	tri.setC(c1, 1);
	tri.setC(c2, 2);
	return tri;
}

void VoxWidget::buildTests()
{
	BoundingBox b(0.f, 0.f, 0.f,
					8.f, 8.f, 8.f);
	m_engine.setBounding(b);
	
	Vector3F vp[4];
	vp[0].set(1.1f, .91f, 1.3f);
	vp[1].set(5.27f, 3.1f, 6.1f);
	vp[2].set(7.49f, 6.721f, 4.9f);
	vp[3].set(4.9f, .2f, .2f);
	
	Vector3F vc[4];
	vc[0].set(1.f, 0.f, 0.f);
	vc[1].set(0.5f, 0.1f, 0.f);
	vc[2].set(0.1f, 0.f, .5f);
	vc[3].set(0.1f, .5f, 0.f);
	
	m_engine.add(createTriangle(vp[0], vp[1], vp[2],
								vc[0], vc[1], vc[2]) );
								
	m_engine.add(createTriangle(vp[0], vp[2], vp[3],
								vc[0], vc[2], vc[3]) );
	
	m_engine.build();
	std::cout<<"\n grid n cell "<<m_engine.numCells();
	std::cout.flush();
	
	b.setMax(4.f, 4.f, 4.f);
	m_engine1.setBounding(b);
	m_engine1.add(createTriangle(vp[0], vp[1], vp[2],
								vc[0], vc[1], vc[2]) );
								
	m_engine1.add(createTriangle(vp[0], vp[2], vp[3],
								vc[0], vc[2], vc[3]) );
	
	m_engine1.build();
	
	b.setMin(4.f, 4.f, 4.f);
	b.setMax(8.f, 8.f, 8.f);
	m_engine2.setBounding(b);
	m_engine2.add(createTriangle(vp[0], vp[1], vp[2],
								vc[0], vc[1], vc[2]) );
								
	m_engine2.add(createTriangle(vp[0], vp[2], vp[3],
								vc[0], vc[2], vc[3]) );
	
	m_engine2.build();
	
	b.setMin(4.f, 4.f, 0.f);
	b.setMax(8.f, 8.f, 4.f);
	m_engine3.setBounding(b);
	m_engine3.add(createTriangle(vp[0], vp[1], vp[2],
								vc[0], vc[1], vc[2]) );
								
	m_engine3.add(createTriangle(vp[0], vp[2], vp[3],
								vc[0], vc[2], vc[3]) );
	
	m_engine3.build();
	
	b.setMin(4.f, 0.f, 0.f);
	b.setMax(8.f, 4.f, 4.f);
	m_engine4.setBounding(b);
	m_engine4.add(createTriangle(vp[0], vp[1], vp[2],
								vc[0], vc[1], vc[2]) );
								
	m_engine4.add(createTriangle(vp[0], vp[2], vp[3],
								vc[0], vc[2], vc[3]) );
	
	m_engine4.build();
	
	b.setMin(4.f, 0.f, 4.f);
	b.setMax(8.f, 4.f, 8.f);
	m_engine5.setBounding(b);
	m_engine5.add(createTriangle(vp[0], vp[1], vp[2],
								vc[0], vc[1], vc[2]) );
								
	m_engine5.add(createTriangle(vp[0], vp[2], vp[3],
								vc[0], vc[2], vc[3]) );
	
	m_engine5.build();
}
